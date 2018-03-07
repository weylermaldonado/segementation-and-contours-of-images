#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <fstream>
#include <fftw3.h>


using namespace cv;
using namespace std;

void guardarMatrizDoble(vector<vector<double> > &matriz);
vector<vector<Point> > extraerContornos();
vector<vector<complex<double> > > convertirContornosEnComplejos(vector<vector<Point> > contornos);
complex<double> convertirCoordenada(int x, int y);
void obtenerDescriptoresFourier(vector <vector<complex<double> > > matrizContornosComplejos);
void calcularDFTInversa(vector<vector<complex <double> > > &matrizContornosComplejos, vector<vector<complex<double> > > &matrizComplejosFrecuencia);
void calcularForwardDFT(vector<vector<complex <double> > > &matrizContornosComplejos, vector<vector<complex<double> > > &matrizComplejosFrecuencia);
void filtrarFrecuencias(vector<vector<complex<double> > > &matrizComplejosFrecuencia);
vector<vector<double> > calcularFase(vector<vector<complex<double> > > &matrizComplejosFrecuencia);
vector<vector<double> > calcularMagnitud(vector<vector<complex<double> > > &matrizComplejosFrecuencia);
void transformarAcomplejos(vector<vector<complex<double> > > &matrizComplejosFrecuencia, vector<vector<double> > &magnitudFrecuencia,vector<vector<double> > &faseFrecuencia);
void iniciarNuevaMatrizCompleja(vector<vector<complex <double> > > &matrizContornosComplejos, vector<vector<complex<double> > > &matrizComplejosFrecuencia);
void imprimirMatrizCompleja(vector<vector<complex<double> > > &matriz);
void imprimirMatrizDoble(vector<vector<double>  > &matriz);
void iniciarNuevaMatrizDoble(vector<vector<complex<double> > >  &matrizContornosComplejos, vector<vector<double> >  &matrizDoble);
void guardarCoordenadasContornoComplejos(vector<vector<complex<double> > > matrizComplejos);
vector<vector<Point> > convertirComplejosEnCoordenadas( vector<vector<complex<double> > > matrizContornosComplejos);
void dibujarContornos(vector<vector<Point> > contornosFiltrados);
void guardarCoordenadasContorno(vector<vector<Point> > contornos);
Mat src, imagenSilueta;
int descriptoresFourier = 0;
vector<Vec4i> hierarchy;

int main( int argc, char** argv ){
  src = imread(argv[1], 1 );
  cvtColor( src, imagenSilueta, CV_BGR2GRAY );
  vector<vector<complex<double> > > matrizComplejos = convertirContornosEnComplejos(extraerContornos());
  obtenerDescriptoresFourier(matrizComplejos);
  convertirComplejosEnCoordenadas(matrizComplejos);
  while(true){
    int c;
    c = waitKey( 20 );
    if( (char)c == 27 ) {
     	break; 
 	 }
   }
}

vector<vector<complex<double> > > convertirContornosEnComplejos(vector<vector<Point> > contornos){
  vector<vector<complex<double> > > vectoresComplejos;
  for(int i = 0; i < contornos.size(); i++){
  	vector<complex<double> > vectorComplejo;
    vectoresComplejos.push_back(vectorComplejo);
    for(int k = 0; k < contornos[i].size(); k++){
       vectoresComplejos.at(i).push_back(convertirCoordenada(contornos[i][k].x, contornos[i][k].y));
    }
  }

  return vectoresComplejos;
}

void dibujarContornos(vector<vector<Point> > contornosFiltrados){
  Mat drawing = Mat::zeros(imagenSilueta.size(),CV_8UC3);
  for( int i = 0; i < contornosFiltrados.size(); i++ ){
  	   Scalar color( rand()&255, rand()&255, rand()&255 );
       drawContours(drawing, contornosFiltrados, i, color, 2, 8, hierarchy, 0, Point() );
  }
  namedWindow("imagen filtrada", CV_WINDOW_AUTOSIZE );
  imshow( "imagen filtrada", drawing);
  imwrite("ImagenFiltrada.jpg" , drawing);
}

vector<vector<Point> > convertirComplejosEnCoordenadas( vector<vector<complex<double> > > matrizContornosComplejos){
	vector<vector<Point> > matrizCoordenadas;
	for(int i = 0; i < matrizContornosComplejos.size(); i++){
		vector<Point>  vectorVacio(matrizContornosComplejos[i].size());
		matrizCoordenadas.push_back(vectorVacio);
		for(int j = 0; j < matrizCoordenadas.at(i).size(); j++){
		   matrizCoordenadas[i][j] = Point(matrizContornosComplejos[i][j].real(), matrizContornosComplejos[i][j].imag());
		}
	}
	guardarCoordenadasContorno(matrizCoordenadas);
	return matrizCoordenadas;
}

void guardarCoordenadasContorno(vector<vector<Point> > contornos){
  ofstream archivoContornos;
  archivoContornos.open("ContornosCartesianos.m");
  for(int i = 0;i < contornos.size(); i++){
     int sizeContorno = contornos[i].size();
    archivoContornos << "Cont" << i << "=[";
    //Guarda abscisas
    for(int k = 0; k < sizeContorno; k++){
      archivoContornos << contornos[i][k].x;
      if(k != sizeContorno - 1){
        archivoContornos << ",";
      }
    }
    archivoContornos << ";";
    //Guarda ordenadas
    for(int k = 0; k < sizeContorno; k++){
      archivoContornos << contornos[i][k].y;
      if(k != sizeContorno - 1){
        archivoContornos << ",";
      }
    }
    archivoContornos << "];" << "\n";
  }
  archivoContornos.close();
}


complex<double> convertirCoordenada(int x, int y){
  double abscisa = x;
  double ordenada = y; 
  double fase, magnitud = 0;
  magnitud = hypot(abscisa,ordenada);
  fase = atan2(ordenada,abscisa);
  return polar(magnitud, fase);
}


void calcularForwardDFT(vector<vector<complex <double> > > &matrizContornosComplejos, vector<vector<complex<double> > > &matrizComplejosFrecuencia){
  fftw_plan plan;
  for(int i = 0; i < matrizContornosComplejos.size(); i++){
  	 int sizeTransformadaSalida =  matrizContornosComplejos[i].size();
  	 plan = fftw_plan_dft_1d(sizeTransformadaSalida,  reinterpret_cast<fftw_complex*>(&matrizContornosComplejos[i].at(0)),  reinterpret_cast<fftw_complex*>(&matrizComplejosFrecuencia[i].at(0)), FFTW_FORWARD, FFTW_ESTIMATE);
  	 fftw_execute(plan);
  	 fftw_destroy_plan(plan);
  }
}

void calcularDFTInversa(vector<vector<complex <double> > > &matrizContornosComplejos, vector<vector<complex<double> > > &matrizComplejosFrecuencia ){
  fftw_plan plan;
  for(int i = 0; i < matrizContornosComplejos.size(); i++){
  	 int sizeTransformadaSalida =  matrizContornosComplejos[i].size();
  	 plan = fftw_plan_dft_1d(sizeTransformadaSalida,reinterpret_cast<fftw_complex*>(&matrizComplejosFrecuencia[i].at(0)) ,reinterpret_cast<fftw_complex*>(&matrizContornosComplejos[i].at(0)), FFTW_FORWARD, FFTW_ESTIMATE);
  	 fftw_execute(plan);
  	 fftw_destroy_plan(plan);
  }
  imprimirMatrizCompleja(matrizComplejosFrecuencia);
}


 vector<vector<double> > calcularMagnitud(vector<vector<complex<double> > > &matrizComplejosFrecuencia){
	vector<vector<double> >  magnitudFrecuencia;
	 iniciarNuevaMatrizDoble(matrizComplejosFrecuencia ,magnitudFrecuencia);
	for(int i = 0; i < magnitudFrecuencia.size(); i++){
		for(int j = 0; j < magnitudFrecuencia[i].size(); j++){
			magnitudFrecuencia[i][j] = abs(matrizComplejosFrecuencia[i][j]);
		}
	}
	return magnitudFrecuencia;
}

 vector<vector<double> > calcularFase(vector<vector<complex<double> > > &matrizComplejosFrecuencia){
	vector<vector<double> > faseFrecuencia;
    iniciarNuevaMatrizDoble(matrizComplejosFrecuencia ,faseFrecuencia);
	for(int i = 0; i < faseFrecuencia.size(); i++){
		for(int j = 0; j < faseFrecuencia[i].size(); j++){
			faseFrecuencia[i][j] = arg(matrizComplejosFrecuencia[i][j]);
		}
	}
	return faseFrecuencia;
}


void transformarAcomplejos(vector<vector<complex<double> > > &matrizComplejosFrecuencia, vector<vector<double> > &magnitudFrecuencia,vector<vector<double> > &faseFrecuencia){
  for(int i = 0; i < matrizComplejosFrecuencia.size(); i++){
  	for(int k = 0; k < matrizComplejosFrecuencia[i].size(); k++){
  		 	matrizComplejosFrecuencia[i][k] = polar(magnitudFrecuencia[i][k], faseFrecuencia[i][k]);
  	}
  }
}

void imprimirMatrizCompleja(vector<vector<complex<double> > > &matriz){
	for(int i = 0; i < matriz.size(); i++){
		for(int k = 0; k < matriz.at(i).size(); k++){
			cout << matriz[i][k]<< " ";
		}
		cout << "\n";
	}
}

void imprimirMatrizDoble(vector<vector<double> > &matriz){
	for(int i = 0; i < matriz.size(); i++){
		for(int k = 0; k < matriz.at(i).size(); k++){
			cout << matriz[i][k]<< " ";
		}
		cout << "\n";
	}
}

void guardarMatrizDoble(vector<vector<double> > &matriz){
    ofstream archivoContornos;
    archivoContornos.open("magnitudesFiltradas.m");
	for(int i = 0; i < matriz.size(); i++){
		for(int k = 0; k < matriz.at(i).size(); k++){
			archivoContornos <<  matriz[i][k] << "," << " ";
		}
		archivoContornos << "\n";
	}
	archivoContornos.close();
}

void guardarCoordenadasContornoComplejos(vector<vector<complex<double> > > matrizComplejos){
  ofstream archivoContornos;
  archivoContornos.open("contornoFiltrado.m");
  for(int i = 0; i < matrizComplejos.size(); i++){
    archivoContornos << "ContC" << i << " = [";
    for(int k = 0; k < matrizComplejos[i].size(); k++){
      archivoContornos << "(" << matrizComplejos[i][k].real() << " + " << matrizComplejos[i][k].imag() << "i" << ")";
      if(k != matrizComplejos[i].size() - 1){
        archivoContornos << ",";
      }
    }
    archivoContornos << "];";
  }
  archivoContornos.close();
}

void filtrarFrecuencias(vector<vector<complex<double> > > &matrizComplejosFrecuencia){
  vector<vector<double> >  magnitudFrecuencia = calcularMagnitud(matrizComplejosFrecuencia);
  vector<vector<double> > faseFrecuencia = calcularFase(matrizComplejosFrecuencia);
  for(int i = 0; i < magnitudFrecuencia.size(); i++){
  	for(int k = 0; k < magnitudFrecuencia[i].size(); k++){
  	 if((k > descriptoresFourier)  && (k < magnitudFrecuencia[i].size() - descriptoresFourier)){	
  			magnitudFrecuencia[i][k] = 0;
  		}
  	}
  }
  guardarMatrizDoble(magnitudFrecuencia);
  transformarAcomplejos(matrizComplejosFrecuencia,magnitudFrecuencia, faseFrecuencia);
  magnitudFrecuencia.clear();
  faseFrecuencia.clear();
}


void iniciarNuevaMatrizCompleja(vector<vector<complex<double> > > &matrizContornosComplejos, vector<vector<complex<double> > > &matrizComplejosFrecuencia){
	for(int i = 0; i < matrizContornosComplejos.size(); i++){
	    vector<complex<double> > vectorVacio(matrizContornosComplejos[i].size());
		matrizComplejosFrecuencia.push_back(vectorVacio);
	}
}

void iniciarNuevaMatrizDoble(vector<vector<complex<double> > >  &matrizContornosComplejos, vector<vector<double> >  &matrizDoble){
	for(int i = 0; i < matrizContornosComplejos.size(); i++){
	    vector<double>  vectorVacio(matrizContornosComplejos[i].size());
		matrizDoble.push_back(vectorVacio);
	}
}

void obtenerDescriptoresFourier(vector <vector<complex<double> > > matrizContornosComplejos){
  cout << "Introducir # de descriptores" << endl;
  cin >> descriptoresFourier;	
  vector <vector<complex<double> > > matrizComplejosFrecuencia;
  iniciarNuevaMatrizCompleja(matrizContornosComplejos, matrizComplejosFrecuencia);
  calcularForwardDFT(matrizContornosComplejos, matrizComplejosFrecuencia);
  filtrarFrecuencias(matrizComplejosFrecuencia);
  calcularDFTInversa(matrizContornosComplejos, matrizComplejosFrecuencia);
}

vector<vector<Point> > extraerContornos(){
  vector<vector<Point> > contornos;
  Canny(imagenSilueta, imagenSilueta, 100, 100*2, 3 );
  findContours(imagenSilueta, contornos,hierarchy, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE, Point(0, 0));
  return contornos;
}