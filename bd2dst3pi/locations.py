import os
repo = os.getenv('ANAROOT')

class loc : pass
#Main project directory
loc.ROOT = repo+'/'
#Output folder
loc.OUT = loc.ROOT+'output/'
#Script folder
loc.SCRIPTS = loc.ROOT+'scripts/'
#Plots, tables, and JSON folder in output
loc.PLOTS = loc.OUT+'plots'
loc.TABLES = loc.OUT+'tables'
loc.JSON = loc.OUT+'json'
#Location of files on CERN EOS filesystem
loc.EOS = '/eos/lhcb/wg/semileptonic/RXcHad/B02Dsttaunu/Run2/ntuples/' #--> previous data

#Location of data and MC with selection cuts applied
loc.DATA = loc.EOS+'/norm/data'
loc.MC = loc.EOS+'/norm/Bd_Dst3pi'
#Location of data and MC with only some selection applied
loc.DATA_STRIP_p = loc.EOS +'/stripped/data' # --> previous data
loc.COMMON = loc.EOS+'/common/data'

loc.DATA_WS_STRIP = loc.EOS+'/stripped/dataWS'
loc.MC_STRIP = loc.EOS+'/stripped/Bd_Dst3pi'
loc.DATA_KPiPi = '/afs/cern.ch/user/d/dhill/public/forAnthony/Bd2DstKPiPi.root'
