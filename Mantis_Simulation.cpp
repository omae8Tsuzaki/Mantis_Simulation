#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
//カマキリの立体視に関する論文の再現
//バターワースフィルタ
void butterworth_lowpass_filter(Mat& filter, double cutoff, int n) {
	CV_Assert(filter.rows % 2 == 0 && filter.cols % 2 == 0);

	int centerX = filter.cols / 2;
	int centerY = filter.rows / 2;

	for (int i = 0; i < filter.rows; i++) {
		for (int j = 0; j < filter.cols; j++) {
			int x = (i + filter.rows / 2) % filter.rows;
			int y = (j + filter.cols / 2) % filter.cols;
			double distance = sqrt(pow(x - centerY, 2) + pow(y - centerX, 2));
			filter.at<float>(i, j) = 1.0 / (1.0 + pow(distance / cutoff, 2 * n));
		}
	}
}
void butterworth(Mat& input, Mat& output, double cutoff, int n, int p_s) {
	Mat complexInput, complexOutput;

	//パディング
	Mat paddedImage;
	copyMakeBorder(input, paddedImage, p_s, p_s, p_s, p_s, BORDER_CONSTANT, Scalar::all(0));

	//フーリエ変換を実行
	dft(paddedImage, complexOutput);

	// フィルタを適用
	Mat b_w_filter(paddedImage.size(), CV_32FC1);
	butterworth_lowpass_filter(b_w_filter, cutoff, n);
	multiply(complexOutput, b_w_filter, complexOutput);

	//フーリエ逆変換を実行
	Mat inverseTransform;
	idft(complexOutput, inverseTransform);

	//実部の画像を取り出して、正規化
	Mat resultImage;
	Mat planes[] = { Mat::zeros(complexOutput.size(), CV_32F), Mat::zeros(complexOutput.size(), CV_32FC1) };
	split(inverseTransform, planes);
	magnitude(planes[0], planes[1], resultImage);
	cv::normalize(resultImage, resultImage, 0, 255, NORM_MINMAX);
	resultImage.convertTo(resultImage, CV_8UC1);
	//切り出し
	output = Mat(resultImage, Rect(p_s, p_s, input.rows, input.cols));
}

//四角い重みのフィルタを作成
//入力画像、出力画像、カーネル(出力)、カーネルサイズ、左右の判定(1が左、0が右)
void Sq_filter(Mat& input, Mat& output, Mat& filter_8U, int ksize, bool mode) {
	//強い興奮性領域の長さ
	int Strong_Excitatory = 8;//59
	//弱い興奮性領域の長さ
	int Weak_Excitatory = 16;//107
	int around_size = (Weak_Excitatory - Strong_Excitatory) / 2;

	CV_Assert(ksize % 2 != 0 && ksize > Weak_Excitatory);

	//受容野の重み
	double a = 0.006777;//0.6777
	double b = 0.00318;//0.318
	double c = -0.000746;//-0.0746

	Point rect_position;
	CV_Assert(mode ==1 || mode == 0);
	if (mode == 1) {//左の場合
		//弱い興奮正領域の左上座標、左カメラのカーネルのため中心が右にずれている。
		rect_position = Point(ksize / 2 - Weak_Excitatory / 2, ksize / 2);
		CV_Assert((rect_position.x + Weak_Excitatory <= input.rows) && (rect_position.y + Weak_Excitatory <= input.cols));
	}
	else {//右の場合
		//弱い興奮正領域の左上座標、左カメラのカーネルのため中心が左にずれている。
		rect_position = Point(ksize / 2 - Weak_Excitatory / 2, ksize / 2 - Weak_Excitatory);
		CV_Assert((rect_position.x + Weak_Excitatory <= input.rows) && (rect_position.y + Weak_Excitatory <= input.cols));
	}

	//カーネル作成
	Mat filter = Mat(Size(ksize, ksize), CV_32FC1);
	filter.forEach<float>([&](float& pixel, const int* position) -> void {
		int j_ = position[0]; // 行インデックス
	    int i_ = position[1]; // 列インデックス
	if ((rect_position.x <= j_) && (j_ < rect_position.x + Weak_Excitatory) && (rect_position.y <= i_) && (i_ < rect_position.y + Weak_Excitatory)) {//重みが右にずれた場合
		if ((rect_position.x + around_size <= j_) && (j_ < rect_position.x + around_size + Strong_Excitatory) && (rect_position.y + around_size <= i_) && (i_ < rect_position.y + around_size + Strong_Excitatory)) {
			//強い興奮性
			pixel = a;//a+c
		}
		else {
			//弱い興奮性
			pixel = b;//b+c
		}
	}
	else {
		// 抑制領域
		pixel = c;//pixel = 0;
	}

	});

	//フィルタ適用
	//入力、出力、出力画像に求めるビット深度、カーネル(浮動小数点)
	filter2D(input, output, -1, filter, Point(-1, -1), 0, BORDER_ISOLATED);//BORDER_REPLICATE -1は入力と出力のビット深度が同じ。
	//filter2D(input, output, CV_32FC1, filter, Point(-1, -1), 0, BORDER_ISOLATED);
	normalize(filter, filter, 0, 255, NORM_MINMAX);
	filter.convertTo(filter_8U, CV_8UC1);
}

//両眼の反応を計算
//左入力画像(CV_32FC1)、右入力画像(CV_32FC1)、視差、出力画像
void Binocular_Response(Mat& Left, Mat& Right, int disparity, float b, float ganma_bino, Mat& output) {
	//結果を格納
	Mat result = Mat(Size(Left.cols, Left.rows), CV_32FC1);

	for (int j = 0; j < Left.rows; j++) {
		for (int i = 0; i < Left.cols; i++) {
			float floor_function = 0;
			int i_R =i - disparity;
			int i_result = i - disparity / 2;//(i + i - disparity) / 2のこと
			if (0 <= i_R && 0 <= i_result) {
				floor_function = std::floor(Left.at<uchar>(j, i) + Right.at<uchar>(j, i_R) + b);
				result.at<float>(j, i_result) = std::pow(floor_function, ganma_bino);
			}
			else {
			}
		}
	}
	normalize(result, result, 0, 255, NORM_MINMAX);
	result.convertTo(output, CV_8UC1);
}

int main(int argc, char *argv[]){
    //image size
	int width = 104;//686
	int height = 104;
	Size SIZE = Size(width, height);
	
	/////////make input image
	Mat input_R = Mat::zeros(SIZE, CV_8UC1);
	Mat input_L = Mat::zeros(SIZE, CV_8UC1);

	/////////Gaussian filter
	Mat Gaussian_L = Mat::zeros(SIZE, CV_8UC1);
	Mat Gaussian_R = Mat::zeros(SIZE, CV_8UC1);
	Mat Gaussian_L_32F, Gaussian_R_32F = Mat(SIZE,CV_32FC1);
	
	/////////IIR filter
	//二層性のIIRフィルタ
	float alpha = 0.8f;
	float alpha_1 = alpha;
	float alpha_2 = 2 * alpha - 1;
	float alpha_3 = 3 * alpha - 2;
	int k1 = -1;
	int k2 = 2;
	int k3 = -1;
	Mat IIR_L_Old_1, IIR_L_Old_2, IIR_L_Old_3, IIR_R_Old_1, IIR_R_Old_2, IIR_R_Old_3 = Mat(SIZE, CV_32FC1);
	Mat IIR_L_1, IIR_L_2, IIR_L_3, IIR_R_1, IIR_R_2, IIR_R_3 = Mat(SIZE, CV_32FC1);
	Mat Bipolar_32F_L, Bipolar_32F_R = Mat(SIZE, CV_32FC1);
	Mat Bipolar_result_L, Bipolar_result_R = Mat(SIZE,CV_8UC1);
	Mat IIR_Result_L_1, IIR_Result_L_2, IIR_Result_L_3, IIR_Result_R_1, IIR_Result_R_2, IIR_Result_R_3 = Mat(SIZE, CV_8UC1);

	//Butterworth filter
	Mat bw_result_L, bw_result_R, bw_result_color_L, bw_result_color_R;
	int p_s = 30;//パディングのサイズ
	double cutoff = 1 / (2 * M_PI * 0.02); // カットオフ周波数  1 / (2 * M_PI * 0.02)
	int n = 1; // フィルタ次数

	//重み付け
	Mat weight_filter_L, weight_filter_R;
	Mat Sq_weight_L, Sq_weight_R, Sq_weight_color_L, Sq_weight_color_R;
	int ksize = 105;//重みカーネルの一辺のサイズ

	//両眼の反応
	int disparity = 8;//2つの円の距離による15
	float b = -0.054f;//論文の数値は-0.054
	float ganma_bino = 5.05f;//論文の数値は5.05
	Mat Response;//両眼の反応の出力画像

	//flame number
	int flame_number = 0;
	int start_circle = 8;//円の開始地点

	while (1) {
		if (flame_number == 200) break;
		//input
		input_R = Scalar(255, 255, 255);
		input_L = Scalar(255, 255, 255);
		//円作成(入力, 中心座標, 半径, 色, 塗りつぶし)
        //Pointのxが画像の範囲外になるとエラーを吐く
		circle(input_R, Point(1 + start_circle + flame_number, 54), 5, Scalar(0, 0, 0), -1);
		circle(input_L, Point(1 + flame_number, 54), 5, Scalar(0, 0, 0), -1);

		//Gaussian filter
		//論文では標準偏差0.7
		GaussianBlur(input_L, Gaussian_L, Size(3, 3), 0.6, 0.6);//17*17
		GaussianBlur(input_R, Gaussian_R, Size(3, 3), 0.6, 0.6);
		Gaussian_L.convertTo(Gaussian_L_32F, CV_32FC1);
		Gaussian_R.convertTo(Gaussian_R_32F, CV_32FC1);

		////////////二層性のIIRフィルタ/////////////
		if (flame_number==0) {
			IIR_L_Old_1 = Gaussian_L_32F.clone();
			IIR_L_Old_2 = Gaussian_L_32F.clone();
			IIR_L_Old_3 = Gaussian_L_32F.clone();
			IIR_R_Old_1 = Gaussian_R_32F.clone();
			IIR_R_Old_2 = Gaussian_R_32F.clone();
			IIR_R_Old_3 = Gaussian_R_32F.clone();
		}
		//--------------L--------------
		IIR_L_1 = alpha_1 * IIR_L_Old_1 + (1 - alpha_1) * Gaussian_L_32F;
		IIR_L_2 = alpha_2 * IIR_L_Old_2 + (1 - alpha_2) * Gaussian_L_32F;
		IIR_L_3 = alpha_3 * IIR_L_Old_3 + (1 - alpha_3) * Gaussian_L_32F;
		IIR_L_Old_1 = IIR_L_1;
		IIR_L_Old_2 = IIR_L_2;
		IIR_L_Old_3 = IIR_L_3;
		IIR_L_1.convertTo(IIR_Result_L_1,CV_8UC1);
		IIR_L_2.convertTo(IIR_Result_L_2, CV_8UC1);
		IIR_L_3.convertTo(IIR_Result_L_3, CV_8UC1);
		Bipolar_32F_L = k1 * IIR_L_1 + k2 * IIR_L_2 + k3 * IIR_L_3;
		//二乗化
		Bipolar_32F_L.forEach<float>([&](float& pixel, const int* position) -> void {
			pixel = static_cast<float>(pixel * pixel);
			});
		Mat Sq_L = Bipolar_32F_L.clone();//バターワースで使用
		cv::normalize(Bipolar_32F_L, Bipolar_32F_L, 0, 255, NORM_MINMAX);
		Bipolar_32F_L.convertTo(Bipolar_result_L,CV_8UC1);

		//---------------R------------------------
		IIR_R_1 = alpha_1 * IIR_R_Old_1 + (1 - alpha_1) * Gaussian_R_32F;
		IIR_R_2 = alpha_2 * IIR_R_Old_2 + (1 - alpha_2) * Gaussian_R_32F;
		IIR_R_3 = alpha_3 * IIR_R_Old_3 + (1 - alpha_3) * Gaussian_R_32F;
		IIR_R_Old_1 = IIR_R_1;
		IIR_R_Old_2 = IIR_R_2;
		IIR_R_Old_3 = IIR_R_3;
		IIR_R_1.convertTo(IIR_Result_R_1, CV_8UC1);
		IIR_R_2.convertTo(IIR_Result_R_2, CV_8UC1);
		IIR_R_3.convertTo(IIR_Result_R_3, CV_8UC1);
		Bipolar_32F_R = k1 * IIR_R_1 + k2 * IIR_R_2 + k3 * IIR_R_3;
		//二乗化
		Bipolar_32F_R.forEach<float>([&](float& pixel, const int* position) -> void {
			pixel = static_cast<float>(pixel * pixel);
			});
		Mat Sq_R = Bipolar_32F_R.clone();
		cv::normalize(Bipolar_32F_R, Bipolar_32F_R, 0, 255, NORM_MINMAX);
		Bipolar_32F_R.convertTo(Bipolar_result_R, CV_8UC1);
		//////////////////////////////////////////////////////

		////////Butterworth filter/////////
		/////L/////
		butterworth(Sq_L, bw_result_L, cutoff, n, p_s);
		cv::applyColorMap(bw_result_L, bw_result_color_L, COLORMAP_JET);
		/////R/////
		butterworth(Sq_R, bw_result_R, cutoff, n, p_s);
		applyColorMap(bw_result_R, bw_result_color_R, COLORMAP_JET);
		////////////////////////////////////////
		
		//////受容野の重み付け////////////////////////////////
		//L
		Mat bw_tmp_L = bw_result_L.clone();
		Sq_filter(bw_tmp_L, Sq_weight_L, weight_filter_L, ksize, 1);
		applyColorMap(Sq_weight_L, Sq_weight_color_L, COLORMAP_JET);

		//R
		Mat bw_tmp_R = bw_result_R.clone();
		Sq_filter(bw_tmp_R, Sq_weight_R, weight_filter_R, ksize, 0);
		applyColorMap(Sq_weight_R, Sq_weight_color_R, COLORMAP_JET);
		///////////////////////////////////////////////////////

		//両眼の反応
		Binocular_Response(Sq_weight_L, Sq_weight_R, disparity, b, ganma_bino, Response);
		//Binocular_Response(Rece_tmp_L, Rece_tmp_R, disparity, b, ganma_bino, Response);
		//cv::resize(Response, Response, cv::Size(), 500.0 / Response.cols, 500.0 / Response.rows);
		Mat Response_color;
		applyColorMap(Response, Response_color, COLORMAP_JET);

		imshow("bw_result_color_L", bw_result_color_L);
		imshow("bw_result_color_R", bw_result_color_R);
		imshow("Sq_weight_color_L", Sq_weight_color_L);
		imshow("Sq_weight_color_R", Sq_weight_color_R);
		imshow("Response_color", Response_color);
		flame_number++;
		if (waitKey(1) == 'q') break;
	}
	cv::destroyAllWindows();
	return 0;
}