print("HELLO")

from bd2dst3pi.definitions import years, magnets
from functions import load_data, create_directory

import BDT


## Variables of the BDT ---------------------------------------------------
part_variables_to_plot = [] # particle,variable

for particle in ['B0', 'Dst', 'tau_pion0', 'tau_pion1', 'tau_pion2']:
    part_variables_to_plot.append((particle,'P'))
    part_variables_to_plot.append((particle,'PT'))

for particle in ['B0', 'Dst', 'tau']:
    part_variables_to_plot.append((particle,'ENDVERTEX_CHI2'))

for particle in ['tau_pion0', 'tau_pion1', 'tau_pion2']:
    part_variables_to_plot.append((particle,'TRACK_CHI2NDOF'))

for particle in ['B0','Dst', 'D0']:
    part_variables_to_plot.append((particle,'M'))

variables = []

for particle,variable in part_variables_to_plot:
    variables.append(f"{particle}_{variable}")

##variables = ['B0_P']
print('variables: ', variables)
## Retrieve data -------------------------------------------------------------
df_tot = {}
df_train = {}


vars = variables
print("retrieve MC")
_, df_train['MC'] = load_data(years,magnets,
                                   type_data = 'MC',vars = variables)
print("retrieve data_strip")
_, df_train['data_strip'] = load_data(years,
                                    magnets,
                                    type_data = 'data_strip',
                                    vars = variables)


# df_train['data_strip'] = df_tot['data_strip']

print("retrieve ws_strip")
_, df_train['ws_strip'] = load_data(years,
                                magnets,type_data = 'ws_strip',
                                vars = variables)


print("cuts on delta(M) and B0_M")
low = 5050.
high = 5550.
for d in 'data_strip','MC','ws_strip':
    df_train[d]["Delta_M"] = df_train[d]["Dst_M"] - df_train[d]["D0_M"]
    df_train[d] = df_train[d].query("Delta_M > 143. and Delta_M < 148.")
    #df_tot[d] = df_tot[d].query(f"B0_M < {high} and B0_M > {low}")
df_tot['data_strip'] = df_train['data_strip'].copy(deep=True)
for d in 'data_strip','MC','ws_strip':    
    for particle in 'B0','Dst', 'D0':
        df_train[d]=df_train[d].drop(f'{particle}_M', 1)
    df_train[d]=df_train[d].drop('Delta_M', 1)
    print(d, df_train[d].columns)

    
variables = variables[:-4]

df_train['ws_strip']=df_train['ws_strip'].sample(frac=0.80)

print("Concatenation")
X, y, df = BDT.concatenate(df_train)
#name_BDT = 'adaboost_without_P_with_cut_deltaM'
name_BDT = 'adaboost_0.8_without_P_cutDeltaM'

print("Print 1D histograms")
BDT.signal_background(df[df.y<0.5], df[df.y>0.5],
                  column=variables,
                  range_column=[
                      #[0,6e5],
                      [0,40000],
                      #[0,4e5],
                      [0,30000],
                      #[0,2e5],
                      [0,10000],
                      #[0,2e5],
                      [0,1e4],
                      #[0,2e5],
                      [0,1e4],
                      [0,100],
                      [0,15],
                      None,None,None,None
                  ],
                  bins=100, figsize = (40,25), name_file = name_BDT, name_folder = name_BDT)

bg,sig = BDT.bg_sig(y)

print("print correlation matrix")
print("Background")
BDT.correlations(df[bg].drop('y', 1),name_file= name_BDT+'_background', name_folder = name_BDT) # Drop the column(->1) 'y'
print("Signal")
BDT.correlations(df[sig].drop('y', 1),name_file= name_BDT+'_signal', name_folder = name_BDT)

## Train the BDT -----------------------------------------------------------
#indices = [list(df.columns).index(e) for e in ('Dst_M','D0_M')]
#X2 = np.delete(X, indices, 1)
#print(len(X))

print("Training of the BDT")
X_train, y_train, X_test, y_test, bdt = BDT.BDT(X, y)


print("Report")
BDT.classification_report_print(X_test, y_test, bdt,name_BDT)
print("ROC curve")
BDT.plot_roc(X_test, y_test, bdt, name_BDT, name_folder = name_BDT)
print("overtraining check")
BDT.compare_train_test(bdt, X_train, y_train, X_test, y_test,name_BDT = name_BDT, name_folder = name_BDT)
print("apply the BDT to our data")
BDT.apply_BDT(df_tot['data_strip'],df_train['data_strip'], bdt,name_BDT=name_BDT)


print("The end")






