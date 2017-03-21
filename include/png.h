#include "lodepng.h"
#include <iostream>
using namespace std;
typedef unsigned char  epngBYTE;

typedef struct RGBApixel {
	epngBYTE Blue;
	epngBYTE Green;
	epngBYTE Red;
	epngBYTE Alpha;
} RGBApixel;

class PNG
{
     private:

     int Width;
     int Height;
     RGBApixel** Pixels;


     public:

     int TellWidth( void );
     int TellHeight( void );

     PNG(const char* input_name);
     ~PNG();
     RGBApixel* operator()(int i,int j);

     RGBApixel GetPixel( int i, int j ) const;

};


