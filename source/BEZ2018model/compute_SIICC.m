function [m,p]=Compute_SIIC (ISI, N)

 m=  mean(ISI);% for each trial
 p = sum((ISI(1:end-1)-m).*(ISI(2:end)-m))*(N-1)/(N-2)...
         /sum((ISI-m).^2);