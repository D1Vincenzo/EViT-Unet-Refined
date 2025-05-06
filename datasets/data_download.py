import synapseclient 
import synapseutils 
from download_token import token

syn = synapseclient.Synapse() 
syn.login(authToken=token) 
# files = synapseutils.syncFromSynapse(syn, 'syn3379050', path='datasets/synapse_data', downloadFile=True)

# Obtain a pointer and download the data
syn3379050 = syn.get(entity='syn3379050', version=1 )

# Get the path to the local copy of the data file
filepath = syn3379050.path