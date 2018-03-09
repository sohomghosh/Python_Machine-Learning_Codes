#Download and install metis
sudo wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
#get updated link from http://glaros.dtc.umn.edu/gkhome/metis/metis/download

tar -xvf metis-5.x.y.tar
cd metis*
make config shared=1
make install


OR
#https://centos.pkgs.org/7/puias-computational-x86_64/metis-5.1.0-12.sdl7.x86_64.rpm.html
sudo wget http://springdale.math.ias.edu/data/puias/computational/7/x86_64//metis-5.1.0-12.sdl7.x86_64.rpm
sudo yum install metis*.rpm

#find location of the following file
libmetis.so.0
#generally it is in /usr/lib64/

sudo vi ~/.bashrc

#add the following line
METIS_DLL=/usr/lib64/libmetis.so.0

source ~/.bashrc

##########Then

pip install metis

#from python console
>import metis
