#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
//カマキリの立体視に関する論文の再現

//DoGフィルタ(全体)
Mat DoG_filter(Mat data, int ksize1, int ksize2, float sigma1, float sigma2) {
	Mat gray_ = data.clone();
	Mat dst_1_, dst_2_;
	Mat dst_3_ = Mat(Size(gray_.cols, gray_.rows), CV_32FC1);
	Mat dst_3_normalize_(Size(gray_.cols, gray_.rows), CV_8UC1);

	GaussianBlur(gray_, dst_1_, Size(ksize1, ksize1), sigma1);
	GaussianBlur(gray_, dst_2_, Size(ksize2, ksize2), sigma2);
	dst_3_ = dst_1_ - dst_2_;

	cv::normalize(dst_3_, dst_3_, 0, 255, NORM_MINMAX);
	dst_3_.convertTo(dst_3_normalize_, CV_8UC1);

	return dst_3_normalize_;
}

Mat Weighting_filter(Mat img, int center_size, int around_size, Point& p_c, Point& p_a) {
	//weight
	double a = 0.677;
	double b = 0.318;
	double  c = -0.0746;
	Mat input_ = Mat(Size(img.cols, img.rows),CV_8UC1);
	input_ = img.clone();
	//Mat tmp_32F_ = Mat::zeros(Size(gray_.cols, gray_.rows), CV_32FC1);
	Mat input_32F_ = Mat(Size(input_.cols, input_.rows), CV_32FC1);
	//Mat tmp_32F_ = Mat(Size(input_.cols, input_.rows), CV_32FC1);
	Mat result_;

	input_.convertTo(input_32F_, CV_32FC1);
	input_32F_.forEach<float>([&](float& pixel, const int* position) -> void {
		int j_ = position[0]; // 行インデックス
		int i_ = position[1]; // 列インデックス
		if ((p_a.x <= j_ )&& (j_ < p_a.x + around_size) && (p_a.y <= i_) && (i_ < p_a.y + around_size)) {
			if ((p_c.x <= j_) && (j_ < p_c.x + center_size) && (p_c.y <= i_) && (i_ < p_c.y + center_size)) {
				// 強い興奮領域
				pixel = a * pixel;
			}
			else {
				// 弱い興奮領域
				pixel = b * pixel;
			}
		}
		else {
			// 抑制領域
			pixel = c * pixel;
		}
	});
	input_32F_.convertTo(result_, CV_8UC1);
	return result_;
}

int main(int argc, char *argv[]){
    //image size
	int width = 400;
	int height = 400;
	
	/////////make input image
	Mat input_R = Mat::zeros(Size(width, height), CV_8UC1);
	Mat input_L = Mat::zeros(Size(width, height), CV_8UC1);

	/////////Gaussian filter
	Mat Gaussian_L = Mat::zeros(Size(width, height), CV_8UC1);
	Mat Gaussian_R = Mat::zeros(Size(width, height), CV_8UC1);
	
	/////////IIR filter
	//時間フィルタに使う最初の画像
	Mat fileter_zero_L = Mat(Size(width, height), CV_32FC1);
	Mat fileter_zero_R = Mat(Size(width, height), CV_32FC1);
	//ガウシアンの結果の型変更
	Mat Gaussian_32F_L = Mat(Size(width, height), CV_32FC1);
	Mat Gaussian_32F_R = Mat(Size(width, height), CV_32FC1);
	//時間フィルタの結果を一時保存
	Mat time_temp_L = Mat(Size(width, height), CV_32FC1);
	Mat time_temp_R = Mat(Size(width, height), CV_32FC1);
	//時間フィルタの結果
	Mat time_L = Mat(Size(width, height), CV_8UC1);
	Mat time_R = Mat(Size(width, height), CV_8UC1);
	//Iirフィルタのα
	float alpha = 0.8f;
	
	//squared
	Mat squared_32F_L = Mat(Size(width, height), CV_32FC1);
	Mat squared_32F_R = Mat(Size(width, height), CV_32FC1);
	Mat squared_L = Mat(Size(width, height), CV_8UC1);
	Mat squared_R = Mat(Size(width, height), CV_8UC1);
	Mat squared_color_L = Mat(Size(width, height), CV_8UC3);
	Mat squared_color_R = Mat(Size(width, height), CV_8UC3);
	
	//DoG filter
	Mat DoG_L = Mat(Size(width, height), CV_8UC1);
	Mat DoG_R = Mat(Size(width, height), CV_8UC1);

	//Box filter
	Mat Weight_L = Mat(Size(width, height), CV_8UC1);
	Mat Weight_R = Mat(Size(width, height), CV_8UC1);
	Point in_L(160,160);//強い興奮領域の左上の座標
	Point out_L(110,110);//弱い興奮領域の左上の座標
	Point in_R(160,160);
	Point out_R(110,110);
	int center_size = 80;//強い興奮領域の一辺の長さ
	int around_size = 180;//弱い興奮領域の一辺の長さ

	//color map
	Mat DoG_L_color = Mat(Size(width, height), CV_8UC3);
	Mat DoG_R_color = Mat(Size(width, height), CV_8UC3);
	int rectsize = 151;//処理する範囲(奇数)
	int ksize = 9;//カーネルサイズ(奇数)//31
	float sigma_1 = 0.6f;//6
	float sigma_2 = 0.9f;//7.5
	if ((rectsize < ksize) || (rectsize % 2 == 0) ) {
		std::cout << "Parameters are wrong." << std::endl;
		return 1;
	}

	//v_L(x_tgtL, y_tgtL), v_R(x_tgtR, y_tgt_R)の格納先
	Mat v_Ltgt_32F = Mat::zeros(cv::Size(width, height), CV_32FC1);
	Mat v_Rtgt_32F = Mat::zeros(cv::Size(width, height), CV_32FC1);
	Mat v_Ltgt = Mat(cv::Size(width, height), CV_8UC1);
	Mat v_Rtgt = Mat(cv::Size(width, height), CV_8UC1);
	Mat v_Ltgt_color = Mat(cv::Size(width, height), CV_8UC3);
	Mat v_Rtgt_color = Mat(cv::Size(width, height), CV_8UC3);

	//床関数
	float b = -0.054f;
	float ganma = 5.05f;
	float R = 0.0f;
	//flame number
	int flame_number = 0;

	while (1) {
		if (flame_number == 200) break;
		//input
		input_R = Scalar(255, 255, 255);
		input_L = Scalar(255, 255, 255);
		//円作成(入力, 中心座標, 半径, 色, 塗りつぶし)
        //Pointのxが画像の範囲外になるとエラーを吐く
		circle(input_R, Point(20 + flame_number, 200), 80, Scalar(0, 0, 0), -1);
		circle(input_L, Point(20 + flame_number, 200), 80, Scalar(0, 0, 0), -1);

		//Gaussian filter
		//論文では標準偏差0.7
		GaussianBlur(input_L, Gaussian_L, Size(3, 3), 0.6, 0.6);//17*17
		GaussianBlur(input_R, Gaussian_R, Size(3, 3), 0.6, 0.6);

		//IIR filter
		Gaussian_L.convertTo(Gaussian_32F_L, CV_32FC1);
		Gaussian_R.convertTo(Gaussian_32F_R, CV_32FC1);
		if (flame_number == 0) {
			fileter_zero_L = Gaussian_32F_L.clone();
			fileter_zero_R = Gaussian_32F_R.clone();
		}
		time_temp_L = (1.0 - alpha) * Gaussian_32F_L + alpha * fileter_zero_L;
		fileter_zero_L = time_temp_L;
		time_temp_L.convertTo(time_L, CV_8UC1);
		
		time_temp_R = (1.0 - alpha) * Gaussian_32F_R + alpha * fileter_zero_R;
		fileter_zero_R = time_temp_R;
		time_temp_R.convertTo(time_R, CV_8UC1);
		
		//二乗、マイナスの値を消すため
		squared_32F_L = time_temp_L.clone();
		squared_32F_L.forEach<float>([](float& pixel_L, const int* posotion_L) -> void {
			pixel_L = static_cast<float>(pixel_L * pixel_L);
			});
		squared_32F_R = time_temp_R.clone();
		squared_32F_R.forEach<float>([](float& pixel_R, const int* posotion_R) -> void {
			pixel_R = static_cast<float>(pixel_R * pixel_R);
			});

		//正規化
		cv::normalize(squared_32F_L, squared_32F_L, 0, 255, cv::NORM_MINMAX);
		cv::normalize(squared_32F_R, squared_32F_R, 0, 255, cv::NORM_MINMAX);
		squared_32F_L.convertTo(squared_L, CV_8UC1);
		squared_32F_R.convertTo(squared_R, CV_8UC1);

		//color map
		//J(x,y,t)
		//applyColorMap(squared_L, squared_color_L, COLORMAP_JET);
		//applyColorMap(squared_R, squared_color_R, COLORMAP_JET);

		//DoG filter
		DoG_L = DoG_filter(squared_L, ksize, ksize, sigma_1, sigma_2);
		DoG_R = DoG_filter(squared_R, ksize, ksize, sigma_1, sigma_2);
		applyColorMap(DoG_L, DoG_L_color, COLORMAP_JET);
		applyColorMap(DoG_R, DoG_R_color, COLORMAP_JET);

		//Box filter
		Weight_L = Weighting_filter(DoG_L, center_size, around_size, in_L, out_L);
		Weight_R = Weighting_filter(DoG_R, center_size, around_size, in_R, out_R);
		
		imshow("DoG_L_color", DoG_L_color);
		imshow("DoG_R_color", DoG_R_color);
		imshow("time_L", time_L);
		imshow("squared_L", squared_L);
		imshow("Weight_L", Weight_L);
		imshow("Weight_R", Weight_R);
		
		flame_number++;
		
		if (waitKey(1) == 'q') break;
	}
	cv::destroyAllWindows();
	return 0;
}