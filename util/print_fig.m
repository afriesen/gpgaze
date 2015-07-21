function print_fig(filename)
    path = './figures/';
%     grid on
    set(gcf,'PaperUnits', 'inches');
    set(gcf,'PaperSize', [10 7.5]);
    set(gcf,'PaperPosition',[0.1 0.1 10 7.5]);
    print('-depsc2','-r300', [path filename '.eps']);
    eps2pdf([path filename '.eps'], 'C:\Program Files\gs\gs8.71\bin\gswin32c.exe');
    delete([path filename '.eps']);
end