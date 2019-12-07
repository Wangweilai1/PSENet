#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "omp.h"
#include <iostream>
#include <algorithm>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace py = pybind11;

static bool sort_by_x(vector<int>box1, vector<int>box2)
{
	return min(box1[0], box1[6]) < min(box2[0], box2[6]);
}


static bool sort_by_y(vector<int>box1, vector<int>box2)
{
	return min(box1[1], box1[7]) < min(box2[1], box2[7]);
}

namespace pse_adaptor {
    void get_kernals(const int *data, vector<long int> data_shape, vector<Mat> &kernals) {
        #pragma omp parallel for
        for (int i = 0; i < data_shape[0]; ++i) {
            Mat kernal = Mat::zeros(data_shape[1], data_shape[2], CV_8UC1);
            for (int x = 0; x < kernal.rows; ++x) {
                for (int y = 0; y < kernal.cols; ++y) {
                    kernal.at<char>(x, y) = data[i * data_shape[1] * data_shape[2] + x * data_shape[2] + y];
                }
            }
            kernals.emplace_back(kernal);
        }
    }

    int growing_text_line(vector<Mat> &kernals, vector<vector<int>> &text_line, float min_area) {
        
        Mat label_mat;
        //int label_num = connectedComponents(kernals[kernals.size() - 1], label_mat, 4);
        Mat stats, centroids;
        int label_num = connectedComponentsWithStats(kernals[kernals.size() - 1], label_mat, stats, centroids, 4);

        // cout << "label num: " << label_num << endl;
        
        int area[label_num + 1];
        memset(area, 0, sizeof(area));
        #pragma omp parallel for
        for (int x = 0; x < label_mat.rows; ++x) {
            for (int y = 0; y < label_mat.cols; ++y) {
                int label = label_mat.at<int>(x, y);
                if (label == 0) continue;
                area[label] += 1;
            }
        }

        queue<Point> queue, next_queue;
        #pragma omp parallel for
        for (int x = 0; x < label_mat.rows; ++x) {
            vector<int> row(label_mat.cols);
            for (int y = 0; y < label_mat.cols; ++y) {
                int label = label_mat.at<int>(x, y);
                
                if (label == 0) continue;
                if (area[label] < min_area) continue;
                
                Point point(x, y);
                queue.push(point);
                row[y] = label;
            }
            text_line.emplace_back(row);
        }

        // cout << "ok" << endl;
        
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        #pragma omp parallel for
        for (int kernal_id = kernals.size() - 2; kernal_id >= 0; --kernal_id) {
            while (!queue.empty()) {
                Point point = queue.front(); queue.pop();
                int x = point.x;
                int y = point.y;
                int label = text_line[x][y];
                // cout << text_line.size() << ' ' << text_line[0].size() << ' ' << x << ' ' << y << endl;

                bool is_edge = true;
                for (int d = 0; d < 4; ++d) {
                    int tmp_x = x + dx[d];
                    int tmp_y = y + dy[d];

                    if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
                    if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
                    if (kernals[kernal_id].at<char>(tmp_x, tmp_y) == 0) continue;
                    if (text_line[tmp_x][tmp_y] > 0) continue;

                    Point point(tmp_x, tmp_y);
                    queue.push(point);
                    text_line[tmp_x][tmp_y] = label;
                    is_edge = false;
                }

                if (is_edge) {
                    next_queue.push(point);
                }
            }
            swap(queue, next_queue);
        }
        return label_num;
    } 

    vector<vector<int>>postprocessing(vector<vector<int>> text_line, int label)
    {
        vector<vector<int>> rectes;
        int H_len = text_line.size();
        int W_len = text_line[0].size();
        if (H_len <= 0 || W_len <= 0 || label <= 1)
            return rectes;
        int lineW = text_line[0].size();
        vector<vector<Point>> vec_rect;
        #pragma omp parallel for
        for(int i = 0; i < label + 1; ++i)
        {
                vector<Point> point;
                vec_rect.push_back(point);
        }
        #pragma omp parallel for
        for(int cy = 0; cy < H_len; ++cy)
        {
            for(int cx = 0; cx < W_len; ++cx)
            {
                if(text_line[cy][cx] == 0)
                    continue;
                Point p = Point(cx, cy);
                vec_rect[text_line[cy][cx]].push_back(p);
            }
        }
        #pragma omp parallel for
        for(int j = 0; j < label + 1; ++j)
        {
            if (vec_rect[j].size() == 0)
                continue;
            Point2f point[4];
            RotatedRect rrect = minAreaRect(vec_rect[j]);
            rrect.points(point);
            //cout << point[1] << "," << point[2] << "," << point[3] << "," << point[0] << endl;
            
            //Rect bbox = r.boundingRect();
            vector<int> rect;
            if ( point[0].x < point[2].x && point[1].x < point[3].x && \
                point[1].y < point[0].y && point[2].y < point[3].y)
            {
                rect.push_back(static_cast<int>(point[1].x));
                rect.push_back(static_cast<int>(point[1].y));
                rect.push_back(static_cast<int>(point[2].x));
                rect.push_back(static_cast<int>(point[2].y));
                rect.push_back(static_cast<int>(point[3].x));
                rect.push_back(static_cast<int>(point[3].y));
                rect.push_back(static_cast<int>(point[0].x));
                rect.push_back(static_cast<int>(point[0].y));
            }
            else if( point[0].x > point[2].x && point[1].x < point[3].x && \
                    point[0].y > point[3].y && point[1].y > point[2].y )
            {
                rect.push_back(static_cast<int>(point[2].x));
                rect.push_back(static_cast<int>(point[2].y));
                rect.push_back(static_cast<int>(point[3].x));
                rect.push_back(static_cast<int>(point[3].y));
                rect.push_back(static_cast<int>(point[0].x));
                rect.push_back(static_cast<int>(point[0].y));
                rect.push_back(static_cast<int>(point[1].x));
                rect.push_back(static_cast<int>(point[1].y));
            }
            else
                continue;
            rectes.push_back(rect);
            //cout << rect[0] << "," << rect[1] << "," << rect[2] << "," << rect[3] << "," << rect[4] <<\
            //    "," << rect[5] << "," << rect[6] << "," << rect[7]  << endl;
            //cout << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << endl;
        }
        return rectes;
    }
	
	vector<vector<int>>boxconnect(vector<vector<int>>__bboxes, float H_Th = 0.7, float  WP_Th = 5, float WN_TH = 40)
	{
		int __N = __bboxes.size();

		set<int> bboxIndex; bboxIndex.clear();
		vector<vector<int>>outBbox; outBbox.clear();

		if (__N < 0){ return outBbox; }
		
		sort(__bboxes.begin(), __bboxes.end(), sort_by_x);

		for (int i = 0; i < __N; i++){
		
			if (bboxIndex.count(i)==0){
				outBbox.push_back(__bboxes[i]);
				bboxIndex.insert(i);
			
				for (int index = 0; index < outBbox.size(); index++)
				{	

					vector<int> bboxes = outBbox[index];
				
					for (int j = i + 1; j<__N;j++ ) 
					{
						if (bboxIndex.count(j) != 0)
							continue;

						int outMinY = min(bboxes[3], __bboxes[j][1]);
						int outMaxY = max(bboxes[5], __bboxes[j][7]);
						int inMinY = max(bboxes[3], __bboxes[j][1]);
						int inMaxY = min(bboxes[5], __bboxes[j][7]);
						int fristRight = min(bboxes[2], bboxes[4]);
						int secondLeft = max(__bboxes[j][0], __bboxes[j][6]);
					
						if (inMaxY - inMinY < 0 || outMaxY - outMinY < 0)
							continue;
						int w_dist = secondLeft - fristRight;

						if ((w_dist < 0 && -1*w_dist > WN_TH) || (w_dist > 0 && w_dist > WP_Th))
							continue;

						if (float(inMaxY - inMinY)/float(outMaxY - outMinY) > H_Th)
						{
							outBbox[index][2] = __bboxes[j][2];
							outBbox[index][3] = __bboxes[j][3];
							outBbox[index][4] = __bboxes[j][4];
							outBbox[index][5] = __bboxes[j][5];
							bboxIndex.insert(j);
							bboxes = outBbox[index];
						}
					}
				}
			}
		}
		sort(outBbox.begin(), outBbox.end(), sort_by_y);
		return outBbox;
	}

    vector<vector<int>> pse(py::array_t<int, py::array::c_style | py::array::forcecast> quad_n9, float min_area) {
        auto buf = quad_n9.request();
        auto data = static_cast<int *>(buf.ptr);
        vector<Mat> kernals;
        get_kernals(data, buf.shape, kernals);

        // cout << "min_area: " << min_area << endl;
        // for (int i = 0; i < kernals.size(); ++i) {
        //     cout << "kernal" << i <<" shape: " << kernals[i].rows << ' ' << kernals[i].cols << endl;
        // }
        
        vector<vector<int>> text_line;
        int label = growing_text_line(kernals, text_line, min_area);

        vector<vector<int>> text_bbox = postprocessing(text_line, label);
        vector<vector<int>> out_text_bbox = boxconnect(text_bbox);
		//for (int i = 0; i < text_bbox.size(); ++i)
		//{
		//	for(int j = 0; j < text_bbox[0].size(); ++j)
		//	{
		//		cout << text_bbox[i][j] << ",";
		//	}
		//	cout << "\n" << endl;
		//}
		//cout << "................................................" << endl;	
		//for (int i = 0; i < out_text_bbox.size(); ++i)
		//{
		//	for(int j = 0; j < out_text_bbox[0].size(); ++j)
		//	{
		//		cout << out_text_bbox[i][j] << ",";
		//	}
		//	cout << "\n" << endl;
		//}
        return out_text_bbox;
    }
}

PYBIND11_PLUGIN(adaptor) {
    py::module m("adaptor", "pse");

    m.def("pse", &pse_adaptor::pse, "pse");

    return m.ptr();
}