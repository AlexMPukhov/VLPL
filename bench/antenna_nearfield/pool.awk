!/^#/ { k=$1; n[k]++; x[k,n[k]]=$2; N[k,n[k]]=$3; el[k,n[k]]=$4; eb[k,n[k]]=$5 }
END {
  for (k in n) if (n[k]>2) dxl = x[k,2]-x[k,1]
  ke = sin(2*3.14159265358979*dxl/0.8)
  for (k in n) { m=n[k]; mx=0
    for (i=2;i<m;i++){ AL[k,i]=sqrt(el[k,i]^2+((el[k,i+1]-el[k,i-1])/(2*ke))^2); AB[k,i]=sqrt(eb[k,i]^2+((eb[k,i+1]-eb[k,i-1])/(2*ke))^2); if(AL[k,i]>mx)mx=AL[k,i] }
    for (i=2;i<m;i++) if (AL[k,i]>0.3*mx) { r=AB[k,i]/AL[k,i]; if (N[k,i]>=4) { fs+=r; fc++ } else { b=int(N[k,i]/bw); s[b]+=r; c[b]++ } } }
  far = fc ? fs/fc : 0
  printf "%s far=%.4f |", name, far
  for (b=0;b<4/bw;b++) printf " %s", c[b] ? sprintf("%.3f", s[b]/c[b]) : "  -  "
  printf "\n"
}
