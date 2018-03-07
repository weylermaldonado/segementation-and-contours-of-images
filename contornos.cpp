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

Mat imagen; Mat imagen_gris;
int min_thresh = 100;
int max_thresh = 100;
RNG rng(12345);






//vector<vector<double> > getMagnitud(vector<vector<complex<double> > > &matrizComplexFrec);
//vector<vector<double> > getFase(vector<vector<complex<double> > > &matrizComplejosFrecuencia);
//void guardarMatriz(vector<vector<double> > &matriz);


void buscaYDibujaContornos(int, void* );
void guardaCoordenadas(vector<vector<Point> > contornos);
void descriptoresFourier(vector <vector<complex<double> > > matrizContornosComplejos, int numDescriptores);
complex<double> convertirCoordenadas(int x, int y);
vector<vector<complex<double> > > contornosComplejos(vector<vector<Point> > contornos);
void convertirAcomplejos(vector<vector<complex<double> > > &matrizComplejosFrecuencia, vector<vector<double> > &magnitudFrecuencia,vector<vector<double> > &faseFrecuencia);
void crearMatrizDoble(vector<vector<complex<double> > >  &matrizContornosComplejos, vector<vector<double> >  &matrizDoble);
void nuevaMatrizCompleja(vector<vector<complex<double> > > &matContornosComplex, vector<vector<complex<double> > > &matComplejosFrecuencia);
void fftwForward(vector<vector<complex <double> > > &matContornosComplex, vector<vector<complex<double> > > &matComplejosFrecuencia);

int main( int argc, char** argv )
{
/*-------------------------------------------Umbralización de la imagen-------------------------------------------------------------*/
  // Leemos la imagen y la convertimso a una escala de grises
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    Mat dst;
    
    // Asignamos un valor máximo y mínimo 
    double thresh = 0;
    double maxValue = 255; 
    
    // Aplicamos threshold
    threshold(src, dst, thresh, maxValue, THRESH_OTSU);

    //mostramos en ventana
    
    namedWindow( "Threshold", WINDOW_AUTOSIZE );
    imshow( "Threshold", dst );     
/*-------------------------------------------Etiquetación y búsqueda de contornos de la imagen-------------------------------------------------------------*/   
  /// leemos una imagen e indicamos que es una imagen de 3 canales
  imagen = imread( argv[1], 1);

  /// convertimos la imagen a gris y le aplicamos un blur
  cvtColor( imagen, imagen_gris, CV_BGR2GRAY );
  blur( imagen_gris, imagen_gris, Size(3,3) );

  
  buscaYDibujaContornos( 0, 0 );

  waitKey(0);
  return(0);
}

void buscaYDibujaContornos(int, void* )
{
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> jerarquia;
  vector<vector<complex<double> > > matrizComplex;
  /// usamos canny para detectar bordes
  Canny( imagen_gris, canny_output, min_thresh, min_thresh*2, 3 );
  /// encontramos contornos
  findContours( canny_output, contours, jerarquia, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );

  ///llamamos método para guardar coordenadas
  guardaCoordenadas(contours);
/*-------------------------------------------Obtención de descriptores de fourier-------------------------------------------------------------*/
  int descriptores = 10;
  matrizComplex = contornosComplejos(contours);
  descriptoresFourier(matrizComplex, descriptores);

  /// dibujamos contornos
  Mat contornos = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
       
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( contornos, contours, i, color, 2, 8, jerarquia, 0, Point() );
     }
 
      
  /// mostramos en ventana
  namedWindow( "Contornos", CV_WINDOW_AUTOSIZE );
  imshow( "Contornos", contornos );
}

///guardamos coordenadas
void guardaCoordenadas(vector<vector<Point> > contornos){
  ofstream file_output;
  file_output.open("contornos_simple.m");
  for(int i = 0;i < contornos.size(); i++){
     int tContorno = contornos[i].size();
    file_output << "Cont" << i << " = [";
    //eje x
    for(int k = 0; k < tContorno; k++){
      file_output << contornos[i][k].x;
      if(k != tContorno - 1){
        file_output << ",";
      }
    }
    file_output << ";";


    //eje y
    for(int k = 0; k < tContorno; k++){
      file_output << contornos[i][k].y;
      if(k != tContorno - 1){
        file_output << ",";
      }
    }
    file_output << "];" << endl;
  }
  file_output.close();
}
///convertimos contornos a números complejos
vector<vector<complex<double> > > contornosComplejos(vector<vector<Point> > contornos){
  vector<vector<complex<double> > > vectoresComplejos;
  for(int i = 0; i < contornos.size(); i++){
  	vector<complex<double> > vectorComplejo;
    vectoresComplejos.push_back(vectorComplejo);
    for(int k = 0; k < contornos[i].size(); k++){
       vectoresComplejos.at(i).push_back(convertirCoordenadas(contornos[i][k].x, contornos[i][k].y));
    }
  }

  return vectoresComplejos;
}
complex<double> convertirCoordenadas(int x, int y){
  double abscisa = x;
  double ordenada = y; 
  double fase, magnitud = 0;
  magnitud = hypot(abscisa,ordenada);
  fase = atan2(ordenada,abscisa);
  return polar(magnitud, fase);
}

/*---------------------------------------------------------------------------------------------*/
void descriptoresFourier(vector <vector<complex<double> > > matContornosComplex, int numDescriptores){
  vector <vector<complex<double> > > matComplejosFrecuencia;
  nuevaMatrizCompleja(matContornosComplex, matComplejosFrecuencia);
  fftwForward(matContornosComplex, matComplejosFrecuencia);
  //filtrarFrecuencias(matrizComplejosFrecuencia);
  //calcularDFTInversa(matrizContornosComplejos, matrizComplejosFrecuencia);
}

void nuevaMatrizCompleja(vector<vector<complex<double> > > &matContornosComplex, vector<vector<complex<double> > > &matComplejosFrecuencia){
	for(int i = 0; i < matContornosComplex.size(); i++){
	    vector<complex<double> > nuevoVector(matContornosComplex[i].size());
		matComplejosFrecuencia.push_back(nuevoVector);
	}
}

void fftwForward(vector<vector<complex <double> > > &matContornosComplex, vector<vector<complex<double> > > &matComplejosFrecuencia){
  fftw_plan plan;
  for(int i = 0; i < matContornosComplex.size(); i++){
  	 int t =  matContornosComplex[i].size();
  	 plan = fftw_plan_dft_1d(t,  reinterpret_cast<fftw_complex*>(&matContornosComplex[i].at(0)),  reinterpret_cast<fftw_complex*>(&matComplejosFrecuencia[i].at(0)), FFTW_FORWARD, FFTW_ESTIMATE);
  	 fftw_execute(plan);
  	 fftw_destroy_plan(plan);
  }
}