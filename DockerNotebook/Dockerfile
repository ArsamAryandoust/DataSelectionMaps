FROM tensorflow/tensorflow:2.5.0-gpu

RUN apt-get update
RUN apt-get upgrade -y

# run this for package opencv-python
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install plotly
RUN pip install kaleido
RUN pip install xlrd
RUN pip install Pillow
RUN pip install numpy
RUN pip install scipy
RUN pip install sklearn
RUN pip install scikit-image
RUN pip install pandas
RUN pip install matplotlib
RUN pip install rasterio
RUN pip install opencv-python

RUN pip install jupyter 

RUN pip install pydot
RUN pip install pydotplus
RUN apt-get install graphviz -y


WORKDIR DataSelectionMaps/src
CMD jupyter notebook --ip 0.0.0.0 --port 1111 --no-browser --allow-root

