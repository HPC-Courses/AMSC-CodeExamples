#include<iostream>
#include <fstream>
#include <cstdlib> // for system()
#include "chrono.hpp"
#undef H5_USE_16_API
#include "hdf5.h"
#include "extendedAssert.hpp"

/*!
  @brief Test for comparing formatted and unformatted io.
  @details It writes and reads an array to from a file using three i/o methods:
  - formatted i/o
  - binary (unformatted) i/o
  - HDF (Hierarchical Data File), an architecture independent binary format.
 */
int main()
{
  using namespace std;
  // Dimension of the array to be stored
  unsigned int const  DIM=10000000;
  // I call system to remove old files
  system("/bin/rm -r -f file.dat");
  system("/bin/rm -r -f file.txt");
  ofstream binfile ("file.dat",ios::binary);
  ofstream asciifile ("file.txt");
  // The array
  double * pippo=new double[DIM];
  for (auto i=0u ;i<DIM; ++i) pippo[i]=static_cast<double>(rand());

  Timings::Chrono tempo;

  cout<<"Writing formatted file"<<endl;
  tempo.start();
  for (auto i=0u ;i<DIM; ++i) asciifile<<pippo[i]<<endl;
  asciifile.flush(); // I want to make sure that everithing is written
  asciifile.close();
  tempo.stop();
  cout<<"Writing done: "<<tempo<<endl;
  
  cout<<"Writing binary file"<<endl;
  tempo.start();
  binfile.write(reinterpret_cast<char* >(pippo),DIM*sizeof(double));
  binfile.flush();
  binfile.close();
  tempo.stop();
  cout<<"Writing done: "<<tempo<<endl;
  
  ifstream binfile2 ("file.dat",ios::binary);
  ifstream asciifile2 ("file.txt");
  cout<<"Reading formatted file"<<endl;
  tempo.start();
  for (auto i=0u ;i<DIM; ++i) asciifile2>>pippo[i];
  asciifile2.close();
  tempo.stop();
  cout<<"Reading done: "<<tempo<<endl;
  
  cout<<"Reading binary file"<<endl;
  tempo.start();
  binfile2.read(reinterpret_cast<char* >(pippo),DIM*sizeof(double));
  binfile2.close();
  tempo.stop();
  cout<<"Reading done: "<<tempo<<endl;


  // Now hdf5 stuff

  // These are the identifiers
  hid_t file_id, dataset_id,dataspace_id;
  // The dataspace definition
  int ndim_dataset=1;
  hsize_t dims=DIM;
  const hsize_t * maxdims(0); //null pointer: fixed size array

  //Create dataset identifier
  dataspace_id=H5Screate_simple(ndim_dataset,&dims,maxdims);
  SURE_ASSERT(dataspace_id>=0,"Cannot create dataspace");
  
  //Create a new file
  // Not necessary to remove file if H5F_ACC_TRUNC flag is used
  // system("/bin/rm -r -f file.h5");
  file_id=H5Fcreate("file.h5",H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
  SURE_ASSERT(file_id>0,"Cannot create file file.h5");
  //Create the dataset
  dataset_id = H5Dcreate(file_id,"/mydata", H5T_NATIVE_DOUBLE,dataspace_id,
             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  SURE_ASSERT(dataset_id>=0,"Cannot create dataset");

  cout<<"Writing HDF file"<<endl;
  herr_t status;
  tempo.start();
  status=H5Dwrite(dataset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL, H5P_DEFAULT,reinterpret_cast<void*>(pippo));
  SURE_ASSERT(status>=0,"Cannot write file. Status"<< status);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
  tempo.stop();
  cout<<"Writing done: "<<tempo<<endl;

  // Open the file read only
  file_id=H5Fopen("file.h5",H5F_ACC_RDONLY,H5P_DEFAULT);
  SURE_ASSERT(file_id>0,"Cannot open file file.h5 read only");
  dataset_id = H5Dopen(file_id,"/mydata", H5P_DEFAULT);

  cout<<"Reading HDF file"<<endl;
  tempo.start();
  status=H5Dread(dataset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL, H5P_DEFAULT,pippo);
  SURE_ASSERT(status>=0,"Cannot read file data. Status"<< status);
  H5Dclose(dataset_id);
  H5Fclose(file_id);
  tempo.stop();
  cout<<"Reading done: "<<tempo<<endl;  


  // Cleanup vector
  delete[] pippo;
}
