#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio/videoio.hpp"

#include <ctype.h>
#include <iostream>

using namespace cv;
using namespace std;

const Scalar blue = Scalar(255, 0, 0);
const Scalar green = Scalar(0, 255, 0);
const Scalar red = Scalar(0, 0, 255);

const float THRESH = 2.0;
const int MAX_COUNT = 500;
Size SUBPIXWINSIZE(10, 10), WINSIZE(31, 31);
TermCriteria TERMCRIT(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.03);

static void help() {
  cout << "\nLukas-Kanade optical flow\n"
       << "OpenCV version: " << CV_VERSION << endl;
  cout << "\tESC - quit the program\n"
       << "\tr - auto-initialize tracking\n"
       << "\tc - delete all the points\n"
       << "\tn - switch the \"dark\" mode on/off\n"
       << "To add/remove a feature point click it\n"
       << endl;
}

static void initFeatures(const Mat &gray_, vector<Point2f> &points_) {
  goodFeaturesToTrack(gray_, points_, MAX_COUNT, 0.01, 10);
  cornerSubPix(gray_, points_, SUBPIXWINSIZE, Size(-1, -1), TERMCRIT);
}

// calculate the optical flow for a sparse feature set using the iterative
// LK method with pyramids - calcOpticalFlowPyrLK
static void opticalFlow(Mat &prevGray_, Mat &gray_, Mat &image_,
                        vector<Point2f> &prevFeatures,
                        vector<Point2f> &features) {
  vector<uchar> status;
  vector<float> err;
  if (prevGray_.empty()) {
    gray_.copyTo(prevGray_);
  }
  calcOpticalFlowPyrLK(prevGray_, gray_, prevFeatures, features, status, err,
                       WINSIZE, 3, TERMCRIT, 0, 0.001);
  // loops through the output vector of 2D points, rearrange the vector such
  // that it only includes those whose flow features were found, mark those on
  // the image, then resize the vector to exclude those that were not found
  size_t i, k;
  if (features.size() != prevFeatures.size()) {
    std::cout << "sizes don't match\n";
  }
  for (i = k = 0; i < features.size(); i++) {
    if (!status[i]) {
      continue;
    }
    features[k] = features[i];
    prevFeatures[k++] = prevFeatures[i];
    circle(image_, features[i], 1, blue, -1);
  }
  features.resize(k);
  prevFeatures.resize(k);

  for (i = 0; i < features.size(); i++) {
    // change in x, change in y
    float diffx = features[i].x - prevFeatures[i].x;  // + if right
    float diffy = features[i].y - prevFeatures[i].y;  // + if downward
    // straight line distance of the change for the flow feature
    float distChange = std::sqrt(std::pow(diffx, 2) + std::pow(diffy, 2));
    // for small change, color the feature blue
    if (distChange < THRESH) {
      circle(image_, features[i], 3, blue, -1);
      continue;
    }

    if (std::abs(diffy) < std::abs(diffx)) {
      // choose color depending on change in x
      if (prevFeatures[i].x < features[i].x) {
        circle(image_, features[i], 3, green, -1);
      } else {
        circle(image_, features[i], 3, red, -1);
      }
    } else {
      // choose color depending on change in y
      if (prevFeatures[i].y < features[i].y) {
        circle(image_, features[i], 3, green, -1);
      } else {
        circle(image_, features[i], 3, red, -1);
      }
    }
  }
}

int main(int argc, char **argv) {
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cout << "error, could not open webcam\n";
    return -1;
  }
  help();
  bool needToInit = false;
  bool darkMode = false;
  namedWindow("motion tracking", 1);
  Mat gray, prevGray, image, frame;
  vector<Point2f> points[2];
  for (;;) {
    cap >> frame;
    if (frame.empty()) {
      break;
    }

    frame.copyTo(image);
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // only the tracking features will be shown on the display image
    if (darkMode) {
      image = Scalar::all(0);
    }

    // if 'r' was hit, initialize the features we want to track
    if (needToInit) {
      initFeatures(gray, points[1]);
    } else if (!points[0].empty()) {
      opticalFlow(prevGray, gray, image, points[0], points[1]);
    }

    // display the image and reset our init flag
    needToInit = false;
    imshow("motion tracking", image);

    char c = (char)waitKey(10);
    if (c == 27) break;
    switch (c) {
      case 'r':
        needToInit = true;
        break;
      case 'c':
        points[0].clear();
        points[1].clear();
        break;
      case 'd':
        darkMode = !darkMode;
        break;
    }

    // swap our current data into the holder for prev in preparation for the
    // next frame
    std::swap(points[1], points[0]);
    cv::swap(prevGray, gray);
  }

  return 0;
}
