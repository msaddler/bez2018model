function [sponts,tabss,trels]=generateANpopulation(numcfs,numsponts)

tabsmax = 1.5*461e-6;
tabsmin = 1.5*139e-6;
trelmax = 894e-6;
trelmin = 131e-6;

% generate sponts, tabss & trels for LS fibers (fiberType = 1)
sponts.LS = min(max(0.1+0.1*randn(numcfs,numsponts(1)),1e-3),0.2);
refrand = rand(numcfs,numsponts(1));
tabss.LS = (tabsmax - tabsmin)*refrand + tabsmin;
trels.LS = (trelmax - trelmin)*refrand + trelmin;

% generate sponts, tabss & trels for MS fibers (fiberType = 2)
sponts.MS = min(max(4+4*randn(numcfs,numsponts(2)),0.2),18);
refrand = rand(numcfs,numsponts(2));
tabss.MS = (tabsmax - tabsmin)*refrand + tabsmin;
trels.MS = (trelmax - trelmin)*refrand + trelmin;

% generate sponts, tabss & trels for HS fibers (fiberType = 3)
sponts.HS = min(max(70+30*randn(numcfs,numsponts(3)),18),180);
refrand = rand(numcfs,numsponts(3));
tabss.HS = (tabsmax - tabsmin)*refrand + tabsmin;
trels.HS = (trelmax - trelmin)*refrand + trelmin;

if exist ('OCTAVE_VERSION', 'builtin') ~= 0
    save('-mat','ANpopulation.mat','sponts','tabss','trels')
else
    save('ANpopulation.mat','sponts','tabss','trels')
end

