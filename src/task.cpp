#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>



#include <iostream>

#include <dirent.h>
#include <string.h>

#include "lodepng.h"
#include "png.h"
#include "classifier.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"


#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

#define RESIZE_X 32
#define RESIZE_Y 32
#define M_HIST_SIZE 16
#define CELL_SIZE 8
#define NAME_MAX 256

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

//typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<PNG*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

typedef Matrix<int> TM_Image;


//add to hystVector features using Local Binary Template
void LBT(TM_Image image, vector<float> &hystVector)
{
    // num of cells
    int cell = 8;

    int imageH = image.n_rows;
    int imageW = image.n_cols;

    //count height and width for cell
    int cell_s_h = trunc(imageH / cell);
    int cell_s_w = trunc(imageW / cell);

    float *result_hist =new float[256];
    for(int tmp = 0 ; tmp<256;tmp++)
	result_hist[tmp] = 0;

    for(int i = 0; i < cell; i++)
            for(int j = 0; j < cell; j++)
            {
                int end_step_x;
		int end_step_y;

		if( i == (cell-1))
			end_step_x = imageW  - cell_s_w*(cell-1);
		else
			end_step_x = cell_s_w;

		if( j == (cell-1))
			end_step_y = imageH  - cell_s_h*(cell-1);
		else
			end_step_y = cell_s_h;

		for(int step_x = 0 ; step_x < end_step_x ; step_x++)
                    for(int step_y = 0 ; step_y < end_step_y ; step_y++)
                    {
			int koordinata_x = step_x + cell_s_w*i;
			int koordinata_y = step_y + cell_s_h*j;
			int num = 0;
                        for(int w = -1;w<2 ; w++)
				for(int h = -1;h<2 ; h++)
				{
				    if((w == 0)&&(h == 0))
					continue;
				    int pos_x = koordinata_x + w;
 				    int pos_y = koordinata_y + h;
				    if(pos_x<0)
					pos_x = 0;
				    if(pos_y<0)
					pos_y = 0;
				    if(pos_x >= imageW)
					pos_x = imageW-1;
				    if(pos_y >= imageH)
					pos_y = imageH-1;
				    if(image(koordinata_y ,koordinata_x )<= image(pos_y ,pos_x ))
				    {
					num = num<<1;
					num++;
				    }
				    else
				    {
					num = num<<1;
				    }

				}
			result_hist[num]++;

                    }
		    //normalize the hyst
		    double hist_norm = 0.0;
                    for(int tmp = 0; tmp < 256 ; tmp++)
                        hist_norm += pow(result_hist[tmp], 2);

                double sq_hist_norm = sqrt(hist_norm);
                if (sq_hist_norm > 0.0001)
                {
                    for(int tmp = 0; tmp < 256 ; tmp++)
                        result_hist[tmp] /= sq_hist_norm;
                }
                for(int tmp = 0; tmp < 256 ; tmp++)
                    hystVector.push_back(result_hist[tmp]);

            }
    return;
}

//add to hystVector features using Color Filter
void ColorFeatures(PNG *image, vector<float> &hystVector)
{
    //num of cells
    int cell = 16;

    int imageH = image->TellHeight();
    int imageW = image->TellWidth();

    //count the height and width for cells
    int cell_s_h = trunc(imageH / cell);
    int cell_s_w = trunc(imageW / cell);




    for(int i = 0; i < cell; i++)
            for(int j = 0; j < cell; j++)
            {
                float sum_r = 0,sum_g = 0,sum_b = 0;
		int end_step_x;
		int end_step_y;

		if( i == (cell-1))
			end_step_x = imageW  - cell_s_w*(cell-1);
		else
			end_step_x = cell_s_w;

		if( j == (cell-1))
			end_step_y = imageH  - cell_s_h*(cell-1);
		else
			end_step_y = cell_s_h;

		//count the average color
		uint pixel_in_cell = 0;
		for(int step_x = 0 ; step_x < end_step_x ; step_x++)
                    for(int step_y = 0 ; step_y < end_step_y ; step_y++)
                    {
                        RGBApixel pixel = image->GetPixel(step_x + cell_s_w*i ,step_y +cell_s_h*j);
			pixel_in_cell++;
                        sum_r += pixel.Red;
                        sum_g += pixel.Green;
                        sum_b += pixel.Blue;
                    }
                sum_r /= pixel_in_cell;
                sum_r /= 255;
                sum_g /= pixel_in_cell;
                sum_g /= 255;
                sum_b /= pixel_in_cell;
                sum_b /= 255;
                hystVector.push_back(sum_r);
		hystVector.push_back(sum_g);
		hystVector.push_back(sum_r);

            }
    return ;

}



//make picture gray for LBT
TM_Image Grayscale(PNG* image)
{
    int imageH = image->TellHeight();
    int imageW = image->TellWidth();
    TM_Image gray_image = TM_Image(imageH, imageW);
    for(int r = 0; r < imageH ; r++ )
        for(int c = 0; c < imageW ; c++ )
        {
            RGBApixel pixel = image->GetPixel(c , r);
            gray_image(r,c) = pixel.Red * 0.299 + pixel.Green * 0.587+ pixel.Blue * 0.114;
            if(gray_image(r , c) >255)
                gray_image(r , c) = 255;
            if(gray_image(r , c) < 0)
                gray_image(r , c) = 0;
        }
    return gray_image;
}

//resize our picture using bicubic functions
TM_Image ResizeIm(TM_Image src_image) {

    int p00;
    int p01;
    int p10;
    int p11;
    int p02;
    int p20;
    int p21;
    int p12;
    int p22;
    int p0_1;
    int p_10;
    int p_1_1;
    int p1_1;
    int p_11;
    int p_12;
    int p2_1;

    double x0, y0;
    double scale_x = static_cast<double>(src_image.n_cols) / RESIZE_X;
    double scale_y = static_cast<double>(src_image.n_rows) / RESIZE_Y;

    TM_Image result_img(RESIZE_Y , RESIZE_X);

    for (int i = 0; i < static_cast<int>(result_img.n_rows); i++)
        for (int j = 0; j < static_cast<int>(result_img.n_cols); j++)
        {
            double y = i * scale_y;
            double x = j * scale_x;
            uint row = floor(y);
            uint col = floor(x);

            if (row == 0) row = 1;
            if (row >= src_image.n_rows - 2)
                row = src_image.n_rows - 3;

            if (col == 0) col = 1;
            if (col >= src_image.n_cols - 2)
                col = src_image.n_cols - 3;

            x0 = x - col;
            y0 = y - row;

            p00 = src_image(row, col);
            p01 = src_image(row, col + 1);
            p10 = src_image(row + 1 , col);
            p11 = src_image(row + 1, col + 1);
            p02 = src_image(row, col + 2);
            p20 = src_image(row + 2, col);
            p21 = src_image(row + 2, col + 1);
            p12 = src_image(row + 1, col + 2);
            p22 = src_image(row + 2, col + 2);
            p0_1 = src_image(row, col - 1);
            p_10 = src_image(row - 1, col);
            p_1_1 = src_image(row - 1, col - 1);
            p1_1 = src_image(row + 1, col - 1);
            p_11 = src_image(row - 1 , col + 1);
            p_12 = src_image(row - 1 , col + 2);
            p2_1 = src_image(row + 2, col - 1);


            //формулы из википедии
            double k1 = (1./4.) * (x0-1)*(x0-2)*(x0+1)*(y0-1)*(y0-2)*(y0+1);
            double k2 = (-1./4.) * (x0)*(x0+1)*(x0-2)*(y0-1)*(y0-2)*(y0+1);
            double k3 = (-1./4.) * (y0)*(x0-1)*(x0-2)*(x0+1)*(y0+1)*(y0-2);
            double k4 = 1./4. * (x0)*(y0)*(x0+1)*(x0-2)*(y0+1)*(y0-2);
            double k5 = (-1./12.) * (x0)*(x0-1)*(x0-2)*(y0-1)*(y0-2)*(y0+1);
            double k6 = (-1./12.) * (y0)*(x0-1)*(x0-2)*(x0+1)*(y0-1)*(y0-2);
            double k7 = 1./12. * (x0)*(y0)*(x0-1)*(x0-2)*(y0+1)*(y0-2);
            double k8 = 1./12. * (x0)*(y0)*(x0+1)*(x0-2)*(y0-1)*(y0-2);
            double k9 = 1./12. * (x0)*(x0-1)*(x0+1)*(y0-1)*(y0-2)*(y0+1);
            double k10 = 1./12. * (y0)*(x0-1)*(x0-2)*(x0+1)*(y0-1)*(y0+1);
            double k11 = 1./36. * (x0)*(y0)*(x0-1)*(x0-2)*(y0-1)*(y0-2);
            double k12 = (-1./12.) * (x0)*(y0)*(x0-1)*(x0+1)*(y0+1)*(y0-2);
            double k13 = (-1./12.) * (x0)*(y0)*(x0+1)*(x0-2)*(y0-1)*(y0+1);
            double k14 = (-1./36.) * (x0)*(y0)*(x0-1)*(x0+1)*(y0-1)*(y0-2);
            double k15 = (-1./36.) * (x0)*(y0)*(x0-1)*(x0-2)*(y0-1)*(y0+1);
            double k16 = (1./36.) * (x0)*(y0)*(x0-1)*(x0+1)*(y0-1)*(y0+1);

            int res_p = round(k1*p00+k2*p01+k3*p10+k4*p11+k5*p0_1+k6*p_10+k7*p1_1+k8*p_11+k9*p02+k10*p20+k11*p_1_1+
            k12*p12+k13*p21+k14*p_12+k15*p2_1+k16*p22);

            res_p = res_p < 0 ? 0 : res_p;
            res_p = res_p > 255 ? 255 : res_p;
            result_img(i, j) = res_p;
        }

    return result_img;
}


//convolve the image with a Sobel filter
TM_Image Convolution(TM_Image src_image, Matrix<double> kernel) {
    int kernel_radX = kernel.n_cols / 2;
    int kernel_radY = kernel.n_rows / 2;

    TM_Image src_image_bord = src_image.extra_borders(kernel_radY , kernel_radX);


    TM_Image dst_image =  TM_Image (src_image.n_rows, src_image.n_cols);


    double sum_ker = 0.0;
    int p;

    for (uint x = kernel_radX; x < src_image_bord.n_cols - kernel_radX; x++) {
        for (uint y = kernel_radY; y < src_image_bord.n_rows - kernel_radY; y++) {
            sum_ker = 0.0;
            for (uint k = 0; k < kernel.n_cols; k++)
                for (uint l = 0; l < kernel.n_rows; l++){
                    p = src_image_bord(y + l - kernel_radY, x + k - kernel_radX);
                    sum_ker += p * kernel(l, k);
                }
            dst_image(y - kernel_radY, x - kernel_radX) = sum_ker;
        }
    }

    return dst_image;
}

//find gradient norm
Matrix<float> GradientNorm(TM_Image horizontal , TM_Image vertical)
{
    Matrix<float> result = Matrix<float>(horizontal.n_rows ,horizontal.n_cols );
    for(uint i = 0 ; i < horizontal.n_rows ; i++)
        for(uint j = 0 ; j < horizontal.n_cols ; j++)
        {
            result(i , j ) = sqrt(pow(horizontal(i , j), 2) + pow(vertical(i , j), 2));
        }
    return result;
}


//find gradient direction
TM_Image GradientDirection(TM_Image horizontal , TM_Image vertical)
{

    TM_Image result = TM_Image(horizontal.n_rows ,horizontal.n_cols );
    for(uint i = 0 ; i < horizontal.n_rows ; i++)
        for(uint j = 0 ; j < horizontal.n_cols ; j++)
        {
            result(i , j ) =  floor(atan2(horizontal(i , j), vertical(i , j)) / (2 * M_PI / M_HIST_SIZE)) + M_HIST_SIZE / 2;
            //int iii = result(i , j );
            if (result(i , j) < 0) result(i , j ) = 0;
            if (result(i , j) >= M_HIST_SIZE ) result(i , j ) = M_HIST_SIZE - 1;
            ///iii = result(i , j );
        }
    return result;
}

//Load pictures for train and test
void LoadFileList(TFileList* file_list, int flag) {

    DIR *dfd;
    struct dirent *dp;
    char filename[NAME_MAX];
    int it = 0;


    //if we need picture for training model
    if(flag == 0)
    {
        string data_path = "../../data/multiclass/train/clocks/";
        strcpy(filename, "../../data/multiclass/train/clocks");
        dfd=opendir("../../data/multiclass/train/clocks");
        while( (dp=readdir(dfd)) != NULL )
        {

            if(dp->d_name[0] == '.')
                continue;
            it++;
            file_list->push_back(make_pair(data_path + dp->d_name, 1));


        }
        closedir(dfd);

        data_path = "../../data/multiclass/train/crocodiles/";
        strcpy(filename, "../../data/multiclass/train/crocodiles");
        dfd=opendir("../../data/multiclass/train/crocodiles");
        while( (dp=readdir(dfd)) != NULL )
        {
            if(dp->d_name[0] == '.')
                continue;
                it++;
            file_list->push_back(make_pair(data_path + dp->d_name, 2));
        }
        closedir(dfd);




    }else
    //if we need picture for testing model
    {
        string data_path = "../../data/multiclass/test/clocks/";
        strcpy(filename, "../../data/multiclass/test/clocks");
        dfd=opendir("../../data/multiclass/test/clocks");
        while( (dp=readdir(dfd)) != NULL )
        {
            if(dp->d_name[0] == '.')
                continue;
            it++;
            file_list->push_back(make_pair(data_path + dp->d_name, 1));
        }
        closedir(dfd);

        data_path = "../../data/multiclass/test/crocodiles/";
        strcpy(filename, "../../data/multiclass/test/crocodiles");
        dfd=opendir("../../data/multiclass/test/crocodiles");
        while( (dp=readdir(dfd)) != NULL )
        {
            if(dp->d_name[0] == '.')
                continue;
            it++;
            file_list->push_back(make_pair(data_path + dp->d_name, 2));
        }
        closedir(dfd);

    }


}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        //PNG* image = new PNG();
        PNG* image = new PNG(file_list[img_idx].first.c_str());
            // Read image from file
        //image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
    {
        stream << file_list[image_idx].first << " ";
        if(labels[image_idx] == 1)
            stream<<"Clocks"<< endl;
        else
            stream<<"Crocodiles"<< endl;
    }
    stream.close();
}

// Exatract features from dataset.
 void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {

    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {

        TM_Image gray_image = Grayscale(data_set[image_idx].first);
        gray_image = ResizeIm(gray_image);

        Matrix<double> kernel_h (1, 3);
        kernel_h(0, 0) = 1;
        kernel_h(0, 1) = 0;
        kernel_h(0, 2) = -1;
        Matrix<double> kernel_w (3, 1);
        kernel_w(0, 0) = 1;
        kernel_w(1, 0) = 0;
        kernel_w(2, 0) = -1;

        TM_Image Sobel_h = Convolution(gray_image, kernel_h);
        TM_Image Sobel_w = Convolution(gray_image, kernel_w);
        Matrix<float> grad_norm = GradientNorm(Sobel_w , Sobel_h);
        TM_Image grad_dir = GradientDirection(Sobel_w , Sobel_h);

        vector<float> hystVector;

        for(int i = 0; i < trunc(RESIZE_X / CELL_SIZE); i++)
            for(int j = 0; j < trunc(RESIZE_Y / CELL_SIZE); j++)
            {
                float * hist_array = new float[M_HIST_SIZE];
                for(int tmp = 0; tmp < (M_HIST_SIZE) ; tmp++)
                    hist_array[tmp] = 0.0;

                for(int step_x = 0 ; step_x < CELL_SIZE ; step_x++)
                    for(int step_y = 0 ; step_y < CELL_SIZE ; step_y++)
                    {
                        hist_array[grad_dir(j * CELL_SIZE + step_y , i * CELL_SIZE + step_x)] += grad_norm(j * CELL_SIZE + step_y , i * CELL_SIZE + step_x);
                    }

                double hist_norm = 0.0;
                for(int tmp = 0; tmp < M_HIST_SIZE ; tmp++)
                    hist_norm += pow(hist_array[tmp], 2);

                double sq_hist_norm = sqrt(hist_norm);
                if (sq_hist_norm > 0.0001)
                {
                    for(int tmp = 0; tmp < M_HIST_SIZE ; tmp++)
                        hist_array[tmp] /= sq_hist_norm;
                }
                for(int tmp = 0; tmp < M_HIST_SIZE ; tmp++)
                    hystVector.push_back(hist_array[tmp]);
            }
	ColorFeatures(data_set[image_idx].first , hystVector);

	LBT(gray_image , hystVector);

        features->push_back(make_pair(hystVector, data_set[image_idx].second));
   }
}
// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data and save trained model
// to 'model_file'
void TrainClassifier(const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList( &file_list, 0);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(&file_list,1);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}
void Single_prediction(char* image_file){

	TDataSet data_set;
	TFeatures features;
	TFileList file_list;
	TLabels labels;

	file_list.push_back(make_pair(image_file, 1));
	// Load images
	LoadImages(file_list, &data_set);
	// Extract features from images
	ExtractFeatures(data_set, &features);

	// Classifier
	TClassifier classifier = TClassifier(TClassifierParams());
	// Trained model
	TModel model;
	// Load model from file
	model.Load("model.txt");
	// Predict images by its features using 'model' and store predictions
	// to 'labels'
	classifier.Predict(features, model, &labels);
    cout << "Predict - " ;
    if(labels[0] == 1)
        cout<<"clocks"<<endl;
    else
        cout<<"crocodiles"<<endl;	// Save predictions
	// SavePredictions(file_list, labels, prediction_file);
	// Clear dataset structure
	ClearDataset(&data_set);

}

int main(int argc, char** argv) {


	if (argc == 2){

		Single_prediction(argv[1]);

	}else
	{
// Command line options parser
        ArgvParser cmd;
// Description of program
        cmd.setIntroductoryDescription("Machine graphics");
// Add other options
		
		cmd.defineOption("model", "Path to file to save or load model",
		ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
		cmd.defineOption("predicted_labels", "Path to file to save prediction results",
		ArgvParser::OptionRequiresValue);
		cmd.defineOption("train", "Train classifier");
		cmd.defineOption("predict", "Predict dataset");

		// Add options aliases
		cmd.defineOptionAlternative("model", "m");
		cmd.defineOptionAlternative("predicted_labels", "l");
		cmd.defineOptionAlternative("train", "t");
		cmd.defineOptionAlternative("predict", "p");

		// Parse options
		int result = cmd.parse(argc, argv);

		// Check for errors or help option
		if (result) {
			cout << cmd.parseErrorDescription(result) << endl;
			return result;
		}

		// Get values
		string model_file = cmd.optionValue("model");
		bool train = cmd.foundOption("train");
		bool predict = cmd.foundOption("predict");

		// If we need to train classifier
		if (train)
			TrainClassifier(model_file);
		// If we need to predict data
		if (predict) {
		// You must declare file to save images
			if (!cmd.foundOption("predicted_labels")) {
			cerr << "Error! Option --predicted_labels not found!" << endl;
			return 1;
		}
		// File to save predictions
		string prediction_file = cmd.optionValue("predicted_labels");
		// Predict data
		PredictData(model_file, prediction_file);
			}
	}
}


