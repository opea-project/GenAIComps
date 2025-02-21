#######################################################################
# Proxy
#######################################################################
export https_proxy=http://proxy-dmz.intel.com:912/
export http_proxy=http://proxy-dmz.intel.com:912/
export no_proxy="intel.com,.intel.com,10.0.0.0/8,192.168.0.0/16,localhost,127.0.0.0/8,134.134.0.0/16,10.54.162.45"
################################################################
# Configure LLM Parameters based on the model selected.  
################################################################
export LLM_ID=${LLM_ID:-"Babelscape/rebel-large"}
export SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
export OVERLAP=${OVERLAP:-"100"}
export MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
export HUGGINGFACEHUB_API_TOKEN=""
export LLM_MODEL_ID=${LLM_ID}
export TGI_PORT=8008
################################################################
### Echo env variables
################################################################
echo "Extractor details"
echo LLM_ID=${LLM_ID}
echo SPAN_LENGTH=${SPAN_LENGTH}
echo OVERLAP=${OVERLAP}
echo MAX_LENGTH=${MAX_LENGTH}
