% ========== start_tab.m =================
function TB = start_tab(A,b,c,B)
% Stellt zu einer gegebenen Matrix A sowie Vektoren b und c das
% Simplex-Starttableau zur Basis B auf

% Dimension der Matrix A bestimmen
[m n] = size(A); 

% Anfangstableau erstellen (incl. 0. Zeile und Spalte)
T = [1, c, 0; zeros(m,1), A, b];

% Matrix zur Basis B berechnen
T_B = [1; zeros(m,1)];
for j = 1:m
    T_B(:,j+1) = [c(B(j)); A(:,B(j))];
end

% Invertieren des Tableaus T_B
inv_T_B = inv(T_B);

% Bestimmen des Simplextableaus
TB = inv_T_B * T;
end

