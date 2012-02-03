% =============== simplex.m ========================
function [EndTab x_opt opt_zfw B] = simplex(A,b,c,B, rule)

% Bestimmen des Start-Tableaus
EndTab = start_tab(A,b,c,B);
opt_zfw = Inf;
x_opt=[];

if (nargin < 5) || strcmp(rule, 'blatt3')
    pivot_rule = @pivot_element;
elseif strcmp(rule, 'bland')
    pivot_rule = @pivot_bland;
else
    error(sprintf('pivot rule "%s" not known', rule));
end
[Z,S] = pivot_rule(EndTab, B);
% Solange eine Pivot-Spalte gefunden werden kann, wird der Algorithmus
% fortgesetzt
while (S~=0)
   if (Z==0)
       % Wenn eine Pivot-Zeile, aber keine Spalte gewaehlt werden konnte, ist
       % das Problem unbeschraenkt
        printf('unbounded!\n')
        x_opt = [];
        opt_zfw = -inf;
        S = 0;
   else
        % Wenn eine Pivotzeile gefunden wurde, fuehre eine Pivot-Operation
        % durch und bestimme das naechste Pivot-Element
        [EndTab, B] = pivot_operation (Z,S,EndTab,B);
        [Z,S] = pivot_rule(EndTab, B);
   end
end

% Konstruieren der Loesung x_opt (falls nicht vorher schon festgestellt
% wurde, dass das Problem unbeschraenkt ist)
if opt_zfw ~= -Inf

    % x_opt mit Nullen initialisieren
    x_opt = zeros(1,length(c));

    % Werte der Basisvariablen eintragen
    for i=1:length(B)
        x_opt(B(i)) = EndTab(i+1,end);
    end

    % Wenn x_opt eine negative Komponente hat, war das Problem unzulaessig
    if sum(x_opt<0) > 0
        x_opt = [];
        opt_zfw = [];
    else
        opt_zfw = -EndTab(1,end);
    end
end
end
