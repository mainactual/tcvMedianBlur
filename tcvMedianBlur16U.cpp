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
#include <vector>
#include <algorithm>

#ifndef TCV_GET_LINE_PTR
#define TCV_GET_LINE_PTR( _Type, _bytes, _stride, _y ) ( reinterpret_cast<_Type*>( (_bytes)+(_y)*(_stride) ) )
#endif

namespace tcv {

class Histogram {
public:
	static const unsigned int umax = (0x1<<16);
	static const unsigned int mask = 0xffff;
	typedef std::vector< int > Vec;

	// defaults to 16
	Histogram( unsigned int N ):m_N( (N==2 || N==4 || N==16 || N==256 || N==umax ) ? N : 16 ),m_count(0)
	{
		m_inv = umax/m_N;
		unsigned int n = m_N;
		while ( 1 )
		{
			m_h.push_back( Vec( n, 0 ) );
			if ( n >= umax )
				break;
			n *= m_N;
		}
	}
	~Histogram(){}

	void insert( unsigned int val ) 
	{
		m_count += 1;
		val &= mask;
		std::vector< Vec >::reverse_iterator it = m_h.rbegin();
		while ( it != m_h.rend() )
		{
			(*it)[ val ] += 1;
			val /= m_N;
			++it;
		}
	}
	void erase( unsigned int val ) 
	{
		m_count -= 1;
		val &= mask;
		std::vector< Vec >::reverse_iterator it = m_h.rbegin();
		while ( it != m_h.rend() )
		{
			(*it)[ val ] -= 1;
			val /= m_N;
			++it;
		}
	}
	unsigned int getmedian( void ) const
	{
		const unsigned int target = m_count/2+1;
		unsigned int index = 0;
		unsigned int c = 0;
		std::vector< Vec >::const_iterator it = m_h.begin();
		while ( 1 )
		{
			while ( c + (*it)[index] < target )
			{
				c += (*it)[index];
				++index;
			}
			if ( ++it == m_h.end() )
				break;
			index *= m_N;
		}
		return index;
	}
	void clear(void)
	{
		m_count = 0;
		for ( std::vector< Vec >::iterator it = m_h.begin();it!=m_h.end();++it )
			memset( &( (*it)[0] ), 0, it->size() * sizeof(int) );
	}
private:
	Histogram();
	Histogram & operator=( const Histogram & rhs );
	Histogram( const Histogram & rhs );

	std::vector< Vec > m_h;
	const unsigned int m_N;
	unsigned int m_inv;
	unsigned int m_count;
};

class MedianFunction_16U : public cv::ParallelLoopBody {
public:
	MedianFunction_16U(
		unsigned char *pDst,
		unsigned int dstStride,
		const unsigned char *pSrc,
		unsigned int srcStride,
		int width,
		int height,
		int R,
		int N):
	  m_pDst(pDst),
	  m_dstStride(dstStride),
	  m_pSrc(const_cast<unsigned char*>(pSrc)),
	  m_srcStride(srcStride),
	  m_width(width),
	  m_height(height),
	  m_R(R>0?R:1),
	  m_N(N)
	{}
	~MedianFunction_16U(){}

	void operator()( const cv::Range & range ) const
	{
		Compute( range.start, range.end );
	}
	int Compute( int _ymin, int _ymax ) const
	{
		int d = sizeof(unsigned short) * m_width;
		if ( !m_pSrc || !m_pDst || m_srcStride < d || m_dstStride < d )
			return 0;
		if ( _ymin < 0 )
			_ymin = 0;
		if ( _ymax > m_height )
			_ymax = m_height;
		int _xmax = m_width;
		d = m_R;
		int ymin = _ymin + d;
		int ymax = _ymax - d;
		int xmin = d+1;
		int xmax = _xmax - d;

		unsigned short * p;
		int x,y;
		std::vector< unsigned short > vec( (2*m_R+1)*(2*m_R+1) );
		Histogram histogram( m_N );
		int _where = 0;
		for (y =_ymin;y<_ymax;++y)
		{
			p = TCV_GET_LINE_PTR( unsigned short, m_pDst, m_dstStride, y );
			if ( y>=ymin && y<ymax )
			{
				for (x=0;x<d;++x)
					*(p+x) = compute_boundscheck( x, y, vec );
				
				FillHistogram( d, y, histogram );
				*(p+d) = histogram.getmedian();
				for (x=xmin;x<xmax;++x)
				{
					UpdateHistogram( x, y, histogram );
					*(p+x) = histogram.getmedian();
				}
				for (x=xmax;x<_xmax;++x)
					*(p+x) = compute_boundscheck( x, y, vec );
			}
			else
			{
				for (x=0;x<_xmax;++x)
					*(p+x) = compute_boundscheck( x, y, vec );
			}
		}
		
		return 1;
	}
protected:
	unsigned short compute_boundscheck( int x, int y, std::vector< unsigned short > & vec ) const
	{
		int xmax = x+m_R;
		if ( xmax >= m_width ) xmax = m_width-1;
		int ymax = y+m_R;
		if ( ymax >= m_height ) ymax = m_height-1;
		int xmin = x-m_R;
		if ( xmin < 0 ) xmin = 0;
		int ymin = y-m_R;
		if ( ymin < 0 ) ymin = 0;
		
		unsigned int u = 0;

		for (int yy=ymin;yy<=ymax;++yy)
		{
			unsigned short * p = TCV_GET_LINE_PTR( unsigned short, m_pSrc, m_srcStride, yy );

			for (int xx=xmin;xx<=xmax;++xx, ++u)
			{
				vec[u]  = *(p+xx);
			}
		}
		// regular median
		std::vector< unsigned short >::iterator it = vec.begin() + u/2;
		std::nth_element( vec.begin(), it, vec.begin()+u );
		return *it;
	}
	void FillHistogram( int x, int y, Histogram & histogram ) const
	{
		histogram.clear();
		const int xmax = x+m_R;
		const int ymax = y+m_R;
		const int xmin = x-m_R;
		const int ymin = y-m_R;
		
		for (int yy=ymin;yy<=ymax;++yy)
		{
			unsigned short * p = TCV_GET_LINE_PTR( unsigned short, m_pSrc, m_srcStride, yy );

			for (int xx=xmin;xx<=xmax;++xx)
			{
				histogram.insert( *(p+xx) );
			}
		}
	}
	void UpdateHistogram( int x, int y, Histogram & histogram ) const
	{
		const int ymax = y+m_R;
		const int ymin = y-m_R;
		const int xx_remove = x-m_R-1;
		const int xx_insert = x+m_R;

		for (int yy=ymin;yy<=ymax;++yy)
		{
			unsigned short * p = TCV_GET_LINE_PTR( unsigned short, m_pSrc, m_srcStride, yy );

			histogram.erase( *( p+xx_remove) );
			histogram.insert( *( p+xx_insert) );
		}
	}

	unsigned char *m_pDst;
	unsigned char *m_pSrc;

	unsigned int m_dstStride;
	unsigned int m_srcStride;
	int m_width;
	int m_height;
	
	int m_R;
	int m_N;

private:
	MedianFunction_16U();
	MedianFunction_16U( const MedianFunction_16U & rhs );
	MedianFunction_16U& operator=(const MedianFunction_16U &rhs);
};

void MedianBlur( const cv::Mat & in, cv::Mat & out, int R )
{
	if ( in.empty() || R<1 )
		CV_Error(CV_StsError,"wrongs parameters");

	if ( out.size() != in.size() || out.type() != in.type() )
		out = cv::Mat( in.size(), in.type() );

	if ( in.type() == CV_16UC1 )
	{
		tcv::MedianFunction_16U median( out.data, out.step.p[0],in.data, in.step.p[0],in.cols, in.rows, R, 16 );
		cv::parallel_for_( cv::Range( 0, in.rows ), median, cv::getNumThreads()-1 );
	}else
	{
		CV_Error(CV_StsError,"Not impl");
	}
}

} // namespace tcv
