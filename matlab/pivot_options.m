% ========== pivot_element.m =================
function z = pivot_options(T, s)
% Bestimmt in einem Simplex-Tableau T, welche Variablen die Basis
% verlassen können, wenn Spalte s eintritt.

[p q] = size(T);
quotients = -1*ones(p,1);
    % Pivot-Spalte s wurde bereits bestimmt, bestimme nun die Pivot-Zeile
    % Initialisieren
    min_quotient = Inf;
      
    for i = 2:p
        if T(i,s+1) > 0
            % Wenn der betreffende Eintrag groesser 0 ist, wird der
            % zugehoerige Quotient gebildet
            quotient = T(i,q)/T(i,s+1);
			quotients(i,1) = quotient;
            if quotient < min_quotient
                % Wenn der gefundene Quotient kleiner als der kleinste bisher
                % gefundene ist, wird dieser ueberschrieben und sich die neue
                % Zeile gemerkt
                min_quotient = quotient;
            end
        end
    end
z = find(quotients == min_quotient) - 1;
