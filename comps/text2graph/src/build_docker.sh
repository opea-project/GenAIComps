docker build --build-arg http_proxy=http://proxy-dmz.intel.com:912/  \
             --build-arg https_proxy=http://proxy-dmz.intel.com:912/ \
	     -f Dockerfile -t saraghava:graph_extractor ../../../
