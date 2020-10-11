import os
repo = os.getenv('ANAROOT')

class loc : pass
loc.ROOT = repo+'/'
loc.OUT = loc.ROOT+'output/'
loc.PLOTS = loc.OUT+'plots'
loc.TABLES = loc.OUT+'tables'
loc.JSON = loc.OUT+'json'
loc.EOS = '/eos/lhcb/wg/semileptonic/RXcHad/B02Dsttaunu/Run2/ntuples/norm'
loc.DATA = loc.EOS+'data'
loc.MC = loc.EOS+'Bd_Dst3pi'
