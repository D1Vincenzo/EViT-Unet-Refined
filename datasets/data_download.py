import synapseclient 
import synapseutils 
from download_token import token

syn = synapseclient.Synapse() 
syn.login(authToken=token) 
files = synapseutils.syncFromSynapse(syn, 'syn3193805', path='datasets/synapse_data', downloadFile=True)