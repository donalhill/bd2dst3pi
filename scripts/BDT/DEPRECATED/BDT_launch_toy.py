print("HELLO")

import BDT
from bd2dst3pi.definitions import years, magnets
from functions import load_data

## Variables of the BDT ---------------------------------------------------
##part_variables_to_plot = [] # particle,variable
##
##for particle in ['B0', 'Dst', 'tau_pion0', 'tau_pion1', 'tau_pion2']:
##    #part_variables_to_plot.append((particle,'P'))
##    part_variables_to_plot.append((particle,'PT'))
##
##for particle in ['B0', 'Dst', 'tau']:
##    part_variables_to_plot.append((particle,'ENDVERTEX_CHI2'))
##
##for particle in ['tau_pion0', 'tau_pion1', 'tau_pion2']:
##    part_variables_to_plot.append((particle,'TRACK_CHI2NDOF'))
##
##variables = []
##for particle, variable in part_variables_to_plot:
##    variables.append(f"{particle}_{variable}")
variables = ['B0_P']
print('variables: ', variables)
## Retrieve data -------------------------------------------------------------
df_tot = {}
df = {}


vars = variables
print("retrieve MC")
df['MC'], df_tot['MC'] = load_data(years,magnets,
                                   type_data = 'MC',vars = variables)
print("retrieve data_strip")
df['data_strip'], df_tot['data_strip'] = load_data(years[:1],
                                                   magnets[:1],
                                                   type_data = 'data_strip',
                                                   vars = variables)
print("retrieve ws_strip")
df['ws_strip'], df_tot['ws_strip'] = load_data(years[:1],
                                               magnets[:1],type_data = 'ws_strip',
                                               vars = variables)
print("Concatenation")
X, y, df = BDT.concatenate(df_tot)
name_BDT = 'toy'

##print("Print 1D histograms")
##BDT.signal_background(df[df.y<0.5], df[df.y>0.5],
##                  column=variables,
##                  range_column=[
##                      #[0,6e5],
##                      [0,40000],
##                      #[0,4e5],
##                      [0,30000],
##                      #[0,2e5],
##                      [0,10000],
##                      #[0,2e5],
##                      [0,1e4],
##                      #[0,2e5],
##                      [0,1e4],
##                      [0,100],
##                      [0,15],
##                      None,None,None,None
##                  ],
##                  bins=100, figsize = (40,25), name_file = name_BDT)

bg,sig = BDT.bg_sig(y)

##print("print correlation matrix")
##print("Background")
##BDT.correlations(df[bg].drop('y', 1),name_file= 'all_data_background') # Drop the column(->1) 'y'
##print("Signal")
##BDT.correlations(df[sig].drop('y', 1),name_file= 'all_data_signal')

## Train the BDT -----------------------------------------------------------

print("Training of the BDT")
X_train, y_train, X_test, y_test, bdt = BDT.BDT(X, y)

print("Report")
BDT.classification_report_print(X_test, y_test, bdt,name_BDT)
print("ROC curve")
BDT.plot_roc(X_test, y_test, bdt, name_BDT)
print("overtraining check")
BDT.compare_train_test(bdt, X_train, y_train, X_test, y_test,name_BDT = name_BDT)

print("apply the BDT to our data")
BDT.apply_BDT(df_tot,bdt,name_BDT=name_BDT)

print("The end")






