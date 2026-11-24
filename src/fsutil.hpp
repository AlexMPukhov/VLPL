// File-system helpers with POSIX calls: mkdir -p, directory listing, resumable text logs. std::filesystem::create_directories segfaults with the g++ 14.1 toolchain
// on the GPU workstation (a three-line program reproduces it, 2026-10-27), so VLPL does not use std::filesystem.
#pragma once
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

inline void make_dirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i < path.size() && path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (::mkdir(part.c_str(), 0755) != 0 && errno != EEXIST)
            throw std::runtime_error("cannot create directory " + part + ": " + std::strerror(errno));
    }
}

// Names of the entries of a directory (without . and ..); empty if it does not exist.
inline std::vector<std::string> list_dir(const std::string& path) {
    std::vector<std::string> v;
    DIR* d = ::opendir(path.c_str());
    if (!d) return v;
    while (dirent* e = ::readdir(d)) {
        const std::string n = e->d_name;
        if (n != "." && n != "..") v.push_back(n);
    }
    ::closedir(d);
    return v;
}

// A text log whose data lines start with the step number. A fresh run (resume_step < 0) or a missing
// file: opened for writing, fresh = true (write the header). A restart at resume_step: the lines of
// later steps (written before the checkpoint's run stopped) are dropped, comment lines kept, and the
// file is opened for appending, so it ends as in an uninterrupted run.
inline FILE* open_log(const std::string& path, long resume_step, bool& fresh) {
    fresh = true;
    if (resume_step >= 0) {
        if (FILE* in = std::fopen(path.c_str(), "r")) {
            std::string keep, line;
            for (int c; (c = std::fgetc(in)) != EOF;) {
                line.push_back(char(c));
                if (c != '\n') continue;
                if (line[0] == '#' || std::strtol(line.c_str(), nullptr, 10) <= resume_step) keep += line;
                line.clear();
            }
            std::fclose(in);
            FILE* out = std::fopen(path.c_str(), "w");
            if (!out) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
            std::fputs(keep.c_str(), out);
            fresh = false;
            return out;
        }
    }
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    return f;
}
