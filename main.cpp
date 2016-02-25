/*=========================================================================
 *
 *  Copyright mainactual
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "tcvMedianBlur16U.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <sstream>

void mainactual( void )
{
	cv::Mat img = cv::imread("image.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	if ( img.type() != CV_16UC1 )
	{
		cv::Mat t;
		img.convertTo( t, CV_16UC1 );
		img = t;
	}
	cv::Mat img2;
	for ( int R = 1; R < 7; ++R )
	{
		tcv::MedianBlur( img, img2, R );

		std::stringstream ss;
		ss << "image" << R << ".jpg";
		cv::imwrite( ss.str(), img2 );
	}
}
int main(void)
{
	try {
		mainactual();
	}catch ( std::exception & e )
	{
		std::cout << e.what() << std::endl;
	}
	return 0;
}
