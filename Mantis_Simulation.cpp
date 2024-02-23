#include <iostream>
#include <opencv2/opencv.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;

//バターワースフィルタ抜き


//四角い重みのフィルタを作成
//入力画像、出力画像、カーネル(出力)、カーネルサイズ
void Sq_filter(Mat& input, Mat& output, Mat& filter_8U, int ksize) {
	//強い興奮性領域の長さ
	int Strong_Excitatory = 52;//104;//52;
	//弱い興奮性領域の長さ
	int Weak_Excitatory = 106;//212;//106;
	int around_size = (Weak_Excitatory - Strong_Excitatory) / 2;

	CV_Assert(ksize % 2 != 0 && ksize > Weak_Excitatory);

	//受容野の重み
	double a = 0.0006777f;//0.6777
	double b = 0.000318f;//0.318
	double c = -0.0000746f / 7;//-0.0746

	Point rect_position;
	rect_position = Point(ksize / 2 - Weak_Excitatory / 2, ksize / 2 - Weak_Excitatory / 2);

	//カーネル作成
	Mat filter = Mat(Size(ksize, ksize), CV_32FC1);
	filter.forEach<float>([&](float& pixel, const int* position) -> void {
		int j_ = position[0]; // 行インデックス
	    int i_ = position[1]; // 列インデックス
	if ((rect_position.x <= j_) && (j_ < rect_position.x + Weak_Excitatory) && (rect_position.y <= i_) && (i_ < rect_position.y + Weak_Excitatory)) {//重みが右にずれた場合
		if ((rect_position.x + around_size <= j_) && (j_ < rect_position.x + around_size + Strong_Excitatory) && (rect_position.y + around_size <= i_) && (i_ < rect_position.y + around_size + Strong_Excitatory)) {
			//強い興奮性
			pixel = a;
		}
		else {
			//弱い興奮性
			pixel = b;
		}
	}
	else {
		// 抑制領域
		pixel = c;
	}
		});

	//フィルタ適用
	//入力、出力、出力画像に求めるビット深度、カーネル(浮動小数点)
	filter2D(input, output, -1, filter, Point(-1, -1), 0, BORDER_ISOLATED);//BORDER_REPLICATE -1は入力と出力のビット深度が同じ。BORDER_ISOLATED
	//filter2D(input, output, CV_32FC1, filter, Point(-1, -1), 0, BORDER_ISOLATED);
	normalize(filter, filter, 0, 255, NORM_MINMAX);
	filter.convertTo(filter_8U, CV_8UC1);
}

//両眼の反応を計算
//左入力画像(CV_32FC1)、右入力画像(CV_32FC1)、視差、出力画像
void Binocular_Response(Mat& Left, Mat& Right, int disparity, float b, float ganma_bino, Mat& output, double& maxPixelValue, double& AllFrame_maxPixelValue, int frame_number) {
	//結果を格納
	Mat result = Mat(Size(Left.cols, Left.rows), CV_32FC1);

	maxPixelValue = 0.0; // 最大の画素値を初期化

	for (int j = 0; j < Left.rows; j++) {
		for (int i = 0; i < Left.cols; i++) {
			float floor_function = 0;
			int i_R = i - disparity;
			int i_result = i - disparity / 2;//(i + i - disparity) / 2のこと
			if (0 <= i_R && 0 <= i_result) {
				floor_function = std::floor(Left.at<uchar>(j, i) + Right.at<uchar>(j, i_R) + b);
				result.at<float>(j, i_result) = std::pow(floor_function, ganma_bino);

				// 最大の画素値を更新
				if (result.at<float>(j, i_result) > maxPixelValue) {
					maxPixelValue = result.at<float>(j, i_result);
				}
			}

			else {
			}

		}
	}
	if (maxPixelValue > AllFrame_maxPixelValue && 20 <= frame_number ) {//全フレームを通して一番大きい画素値を更新。
		AllFrame_maxPixelValue = maxPixelValue;
	}

	result.convertTo(output, CV_8UC1, 255.0 / (2.40725e+12));//4.66579e+13, 1.92E+13
	//通常の正規化
	// normalize(result, result, 0, 255, NORM_MINMAX);
	// result.convertTo(output, CV_8UC1);
}


int main() {
	Size SIZE = Size(678, 678);
	int frame_number = 0;
	int start_circle = 76;//円の開始地点8,100
	std::string Space = std::to_string(start_circle);//円の間隔
	int radius = 25;//円の半径：10, 20,36,55,83, 
	std::string Diameter = std::to_string(2*radius);//円の直径を文字列に変換

	////視差の定義/////
	double I_eye_to_eye = 0.007;//カマキリの眼間距離7mm→0.007[m]
	double P = 0.021f;//[m]
	float S = 0.1;//カマキリからスクリーンまでの距離[m]
	double degree_to_pixel = 0.154f;// [°/pixel]
	double alpha_screen = 2*std::atan((0.5*P)/S);//[rad]
	double Distance = (I_eye_to_eye *S) / (I_eye_to_eye + 2*S*std::tan(alpha_screen/2));//[rad]
	double delta = 2*std::atan(0.5* I_eye_to_eye / Distance);//[rad]
	
	double disparity_ = (delta*(180/M_PI)) /degree_to_pixel;//視差
	int disparity = int(round(disparity_));//四捨五入
	//int disparity = 78;
	std::string str_disparity = std::to_string(disparity);//文字列に変換
	////////////////

	/////////入力画像作成
	Mat input_R = Mat::zeros(SIZE, CV_8UC1);
	Mat input_L = Mat::zeros(SIZE, CV_8UC1);
	/////////ガウシアンフィルタ
	Mat Gaussian_L, Gaussian_R = Mat::zeros(SIZE, CV_8UC1);
	Mat Gaussian_L_32F, Gaussian_R_32F = Mat(SIZE, CV_32FC1);

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
	Mat Bipolar_32F_tmp_L, Bipolar_32F_tmp_R = Mat(SIZE, CV_32FC1);
	Mat Bipolar_result_L, Bipolar_result_R = Mat(SIZE, CV_8UC1);
	Mat IIR_Result_L_1, IIR_Result_L_2, IIR_Result_L_3, IIR_Result_R_1, IIR_Result_R_2, IIR_Result_R_3 = Mat(SIZE, CV_8UC1);


	//二乗
	Mat Sq_result_L, Sq_result_R = Mat(SIZE, CV_8UC1);


	//重み付け
	Mat weight_filter_L, weight_filter_R;
	Mat Sq_weight_32F_L, Sq_weight_32F_R;
	Mat Sq_weight_L, Sq_weight_R, Sq_weight_color_L, Sq_weight_color_R;
	int ksize = 679;//重みカーネルの一辺のサイズ


	//両眼の反応
	//int disparity = 6;//2つの円の距離による8,100,20
	float b = -0.054f;//論文の数値は-0.054
	float ganma_bino = 5.05f;//論文の数値は5.05
	Mat Response;//両眼の反応の出力画像
	double maxPixelValue;//各フレームの画素内の最大の値
	double AllFrame_maxPixelValue = 0.0;//全フレーム内の最大の画素値

	//動画保存
	int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
	double fps = 30.0;
	VideoWriter writer_L, writer_R;//入力
	VideoWriter writer_G_L, writer_G_R;//ガウシアン
	VideoWriter writer_IIR_L, writer_IIR_R;//二相性のIIR
	VideoWriter writer_Sq_L, writer_Sq_R;//二乗
	VideoWriter writer_weight_L, writer_weight_R;//重み
	VideoWriter writer_Bino;//両眼の反応

	//保存先
    //受容野のサイズ変更
	std::string file_path = "C:/Users/t-tsu/Documents/PROGRAM_FILE/DATA/20231222_Sim/receptive_field/Bipolar/52_106_679/";
	std::string clonepath_L = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_f30_L.mp4";
	std::string clonepath_R = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_f30_R.mp4";
	std::string clonepath_G_L = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_f30_Gs27sig4545_L.mp4";
	std::string clonepath_G_R = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_f30_Gs27sig4545_R.mp4";
	std::string clonepath_IIR_L = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_L.mp4";
	std::string clonepath_IIR_R = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_R.mp4";
	std::string clonepath_Sq_L = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_Sq_L.mp4";
	std::string clonepath_Sq_R = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_Sq_R.mp4";
	std::string clonepath_weight_L = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_Weight_L.mp4";
	std::string clonepath_weight_R = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_Weight_R.mp4";
	std::string clonepath_Bimo = file_path + "Dia" + Diameter + "pixel/R" + Diameter + "_Iir08_Bino.mp4";

	writer_L.open(clonepath_L, fourcc, fps, SIZE, false);
	writer_R.open(clonepath_R, fourcc, fps, SIZE, false);
	writer_G_L.open(clonepath_G_L, fourcc, fps, SIZE, false);
	writer_G_R.open(clonepath_G_R, fourcc, fps, SIZE, false);
	writer_IIR_L.open(clonepath_IIR_L, fourcc, fps, SIZE, false);
	writer_IIR_R.open(clonepath_IIR_R, fourcc, fps, SIZE, false);
	writer_Sq_L.open(clonepath_Sq_L, fourcc, fps, SIZE, false);
	writer_Sq_R.open(clonepath_Sq_R, fourcc, fps, SIZE, false);
	writer_weight_L.open(clonepath_weight_L, fourcc, fps, SIZE, false);
	writer_weight_R.open(clonepath_weight_R, fourcc, fps, SIZE, false);
	writer_Bino.open(clonepath_Bimo, fourcc, fps, SIZE, true);

	if (!writer_L.isOpened() || !writer_R.isOpened()) {                         //動画ファイル生成の確認
		std::cout << "writer Not Opened!" << std::endl;        //生成できてない場合はエラー
		return 1;                                    //プログラム終了
	}
	if (!writer_G_L.isOpened() || !writer_G_R.isOpened()) {
		std::cout << "writer Not Opened!" << std::endl;
		return 2;
	}
	if (!writer_IIR_L.isOpened() || !writer_IIR_R.isOpened()) {
		std::cout << "writer Not Opened!" << std::endl;
		return 3;
	}
	if (!writer_Sq_L.isOpened() || !writer_Sq_R.isOpened()) { 
		std::cout << "writer Not Opened!" << std::endl; 
		return 4;                                    
	}
	if (!writer_weight_L.isOpened() || !writer_weight_R.isOpened()) {
		std::cout << "writer Not Opened!" << std::endl;
		return 5;
	}
	if (!writer_Bino.isOpened()) {
		std::cout << "writer Not Opened!" << std::endl;
		return 6;
	}

	while (1) {
		if (frame_number == 250) break;
		input_R = Scalar(255, 255, 255);
		input_L = Scalar(255, 255, 255);
		//円作成(入力, 中心座標, 半径, 色, 塗りつぶし)
		//直径72,109,165
		circle(input_L, Point(200 + start_circle + frame_number, 678 / 2), radius, Scalar(0, 0, 0), -1);//36,55,83
		circle(input_R, Point(200 + frame_number, 678 / 2), radius, Scalar(0, 0, 0), -1);

		//ガウシアン
		GaussianBlur(input_L, Gaussian_L, Size(27, 27), 4.545, 4.545);//Size(3, 3), 1.5, 1.5
		GaussianBlur(input_R, Gaussian_R, Size(27, 27), 4.545, 4.545);//Size(3, 3), 1.5, 1.5
		Gaussian_L.convertTo(Gaussian_L_32F, CV_32FC1);
		Gaussian_R.convertTo(Gaussian_R_32F, CV_32FC1);

		////////////二層性のIIRフィルタ/////////////
		if (frame_number == 0) {
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
		IIR_L_1.convertTo(IIR_Result_L_1, CV_8UC1);
		IIR_L_2.convertTo(IIR_Result_L_2, CV_8UC1);
		IIR_L_3.convertTo(IIR_Result_L_3, CV_8UC1);
		Bipolar_32F_L = k1 * IIR_L_1 + k2 * IIR_L_2 + k3 * IIR_L_3;
		Bipolar_32F_tmp_L = Bipolar_32F_L.clone();
		cv::normalize(Bipolar_32F_tmp_L, Bipolar_32F_tmp_L, 0, 255, NORM_MINMAX);
		Bipolar_32F_tmp_L.convertTo(Bipolar_result_L, CV_8UC1);

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
		Bipolar_32F_tmp_R = Bipolar_32F_R.clone();
		cv::normalize(Bipolar_32F_tmp_R, Bipolar_32F_tmp_R, 0, 255, NORM_MINMAX);
		Bipolar_32F_tmp_R.convertTo(Bipolar_result_R, CV_8UC1);
		///////////////////////////////////////////////////


		//////二乗
		//-----------L-------------------------
		Bipolar_32F_L.forEach<float>([&](float& pixel, const int* posotion) -> void {
			pixel = static_cast<float>(pixel * pixel);
			});
		Mat Sq_L = Bipolar_32F_L.clone();//バターワースで使用
		cv::normalize(Bipolar_32F_L, Bipolar_32F_L, 0, 255, NORM_MINMAX);
		Bipolar_32F_L.convertTo(Sq_result_L, CV_8UC1);
		Mat Sq_result_L_color;
		cv::applyColorMap(Sq_result_L, Sq_result_L_color, COLORMAP_JET);

		//-----------L-------------------------
		Bipolar_32F_R.forEach<float>([&](float& pixel, const int* posotion) -> void {
			pixel = static_cast<float>(pixel * pixel);
			});
		Mat Sq_R = Bipolar_32F_R.clone();
		cv::normalize(Bipolar_32F_R, Bipolar_32F_R, 0, 255, NORM_MINMAX);
		Bipolar_32F_R.convertTo(Sq_result_R, CV_8UC1);
		Mat Sq_result_R_color;
		cv::applyColorMap(Sq_result_R, Sq_result_R_color, COLORMAP_JET);


		//受容野の重み付け
		//-----------L---------------------
		Mat bw_tmp_L = Sq_result_L.clone();
		Sq_filter(bw_tmp_L, Sq_weight_L, weight_filter_L, ksize);
		Mat Sq_weight_L_color;
		cv::applyColorMap(Sq_weight_L, Sq_weight_L_color, COLORMAP_JET);

		//-----------R---------------------
		Mat bw_tmp_R = Sq_result_R.clone();
		Sq_filter(bw_tmp_R, Sq_weight_R, weight_filter_R, ksize);
		Mat Sq_weight_R_color;
		cv::applyColorMap(Sq_weight_R, Sq_weight_R_color, COLORMAP_JET);

		//両眼の反応
		Binocular_Response(Sq_weight_L, Sq_weight_R, disparity, b, ganma_bino, Response, maxPixelValue, AllFrame_maxPixelValue, frame_number);
		//Binocular_Response(Rece_tmp_L, Rece_tmp_R, disparity, b, ganma_bino, Response);
		std::cout << "Largest pixel value in the current frame:" << std::fixed << maxPixelValue << ", Largest pixel value to date:" << AllFrame_maxPixelValue<< std::endl;
		Mat Response_color;
		cv::applyColorMap(Response, Response_color, COLORMAP_JET);


		cv::imshow("Sq_result_L_color", Sq_result_L_color);
		cv::imshow("Sq_weight_L", Sq_weight_L_color);
		cv::imshow("Response_color", Response_color);

		std::cout << "alpha:" << alpha_screen << ",delta:" << disparity_ << "," << disparity << std::endl;

		//動画格納
		writer_L << input_L;
		writer_R << input_R;
		writer_G_L << Gaussian_L;
		writer_G_R << Gaussian_R;
		writer_IIR_L << Bipolar_result_L;
		writer_IIR_R << Bipolar_result_R;
		writer_Sq_L << Sq_result_L_color;
		writer_Sq_R << Sq_result_R_color;
		writer_weight_L << Sq_weight_L_color;
		writer_weight_R << Sq_weight_R_color;
		writer_Bino << Response_color;

		frame_number++;
		if (waitKey(1) == 'q') break;
	}
	//cv::imwrite("C:/Users/t-tsu/Documents/PROGRAM_FILE/DATA/20231215_Sim/filter.jpg", weight_filter_L);
	cv::destroyAllWindows();
	return 0;
}