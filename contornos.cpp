#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;

Mat imagen; Mat imagen_gris;
int min_thresh = 20;
int max_thresh = 255;
RNG rng(12345);


void buscaYDibujaContornos(int, void* );
void guardaCoordenadas(vector<vector<Point> > contornos);

int main( int argc, char** argv )
{
/*-------------------------------------------Umbralización de la imagen-------------------------------------------------------------*/
  // Leemos la imagen y la convertimso a una escala de grises
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    Mat dst;
    
    // Asignamos un valor máximo y mínimo de acuerdo a la paleta de colores rgb
    double thresh = 0;
    double maxValue = 255; 
    
    // Aplicamos threshold
    threshold(src,dst, thresh, maxValue, THRESH_OTSU);

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

  /// usamos canny para detectar bordes
  Canny( imagen_gris, canny_output, min_thresh, min_thresh*2, 3 );
  /// encontramos contornos
  findContours( canny_output, contours, jerarquia, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0) );

  ///llamamos método para guardar coordenadas
  guardaCoordenadas(contours);
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
