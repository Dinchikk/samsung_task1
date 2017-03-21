
#include <iostream>
#include "lodepng.h"
#include "png.h"



PNG::~PNG()
{
 int i;
 for(i=0;i<Width;i++)
 { delete [] Pixels[i]; }
 delete [] Pixels;

}

RGBApixel* PNG::operator()(int i, int j)
{
 using namespace std;
 if( i >= Width )
 { i = Width-1; }
 if( i < 0 )
 { i = 0;  }
 if( j >= Height )
 { j = Height-1; }
 if( j < 0 )
 { j = 0; }

 return &(Pixels[i][j]);
}


int PNG::TellHeight( void )
{ return Height; }


int PNG::TellWidth( void )
{ return Width; }


PNG::PNG(const char* input_name )
{

    unsigned width_, height_;


    std::vector<unsigned char> image;
    unsigned error = lodepng::decode(image, width_, height_, input_name, LCT_RGB, 8);


    Width = width_;
    Height = height_;


    Pixels = new RGBApixel* [Width];
    for(int i=0 ; i<Width ;i++ )
        Pixels[i] = new RGBApixel[Height];

    if(error)
    {
        std::cout << "error " << error << ": " << lodepng_error_text(error) << std::endl;
    }



    RGBApixel new_p;
    Pixels[4][5]=new_p;


    for( int j=0; j < Height ; j++ )
    {
        for( int i=0; i < Width ; i++ )
        {
            RGBApixel new_pixel;

            new_pixel.Red   = image[3*(width_ *j + i)];

            new_pixel.Green = image[3*(width_ *j + i) + 1];

            new_pixel.Blue  = image[3*(width_ *j + i) + 2];

            new_pixel.Alpha = 0;
            Pixels[i][j] = new_pixel;
        }
    }
}

RGBApixel PNG::GetPixel( int i, int j ) const
{
 using namespace std;
 if( i >= Width )
 { i = Width-1; }
 if( i < 0 )
 { i = 0; }
 if( j >= Height )
 { j = Height-1;  }
 if( j < 0 )
 { j = 0; }

 return Pixels[i][j];
}
