% =========== pivot_bland.m =========================
function [z s] = pivot_bland( T,B )

% Positionen aller negativen red. Kosten bestimmen
red_kost_neg = T(1,2:end-1) < -eps;

% Falls es keine solchen gibt, ist das Tableau optimal
if sum(red_kost_neg) < 1
    z = 0; s = 0;
else
    
% nach Blands Pivotregel den kleinsten Index waehlen fuer den die red. Kosten neg sind
    s = find(red_kost_neg >= 1,1,'first');
   
% Problem ist unbeschraenkt, falls in der Piv. Spalte kein pos. Wert existiert
    if max(T(2:end,1+s)) <= 0
        z = 0;

% Um die Pivotzeile zu bestimmen:
    else
        ratio_vek = T(2:end,end)./T(2:end,1+s);
        for i = 1:length(ratio_vek)
            if T(i+1,s+1) <= 0
                ratio_vek(i) = Inf;
            end
        end

% Nach Blands Pivotregel Pivotzeile bestimmen:
        n = min(ratio_vek);
        tt = find(ratio_vek == n); % tt: Zeilenindizes denen Minimum angenommen wird
        [temp, ZZ] = min(B(tt)); % ZZ: kleinstes i s.d. für B(i) Minimum ang. wird
        z = tt(ZZ); % z: welche Zeile entspricht dieser Basisvariable
    end
end
