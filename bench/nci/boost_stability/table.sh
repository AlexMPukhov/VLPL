#!/bin/sh
# Table of run_linear.sh (README.md here): W_field / K_e at 10, 100, 200, 300 / omega_p and its maximum, the largest e-folding rate
# of the field energy over 15 - 20 / omega_p windows after 10 / omega_p (omega_p), the relative change of W_total, max |dGauss|.
# K_e = W_kin(0) / 1837 (electrons and protons at the same gamma). omega_p of n0 = 0.004 n_c (0.002 for the gamma 20 cases).
O=${1:-bs_linear}
printf "%-34s %8s %8s %8s %8s %8s %7s %9s %8s\n" case 10/wp 100/wp 200/wp 300/wp max rate dWtot gauss
for d in $O/*/; do
  n=$(basename $d); [ -f $d/diag.txt ] || continue
  case $n in *_g20_*) wp=0.044721;; *) wp=0.063246;; esac
  awk -v n=$n -v wp=$wp '!/^#/ && NF > 5 { k++; t[k] = $2 * wp; w[k] = $3; wt[k] = $5; if (k0 == 0) k0 = $4 / 1837; if ($6 > g) g = $6 }
    END { if (k < 3) { printf "%-34s incomplete\n", n; exit }
      for (j = 1; j <= 3; j++) { T = 100 * j; v[j] = "-"; for (i = 1; i <= k; i++) if (t[i] <= T + 1e-9) v[j] = sprintf("%.1e", w[i] / k0) }
      m = 0; for (i = 1; i <= k; i++) if (w[i] > m) m = w[i]
      w10 = 0; for (i = 1; i <= k; i++) if (t[i] <= 10) w10 = w[i]
      r = -1; j = 1
      for (i = 1; i <= k; i++) { while (t[i] - t[j] > 20) j++; if (t[j] >= 10 && t[i] - t[j] > 15 && w[i] > 0 && w[j] > 0) { q = log(w[i] / w[j]) / (t[i] - t[j]); if (q > r) r = q } }
      printf "%-34s %8.1e %8s %8s %8s %8.1e %7.3f %9.1e %8.1e%s\n", n, w10 / k0, v[1], v[2], v[3], m / k0, r / 2, wt[k] / wt[1] - 1, g, (t[k] < 299 ? "  (t " sprintf("%.0f", t[k]) "/wp)" : "") }' $d/diag.txt
done
