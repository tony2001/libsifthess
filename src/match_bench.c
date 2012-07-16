/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
//#define NN_SQ_DIST_RATIO_THR 0.49
#define NN_SQ_DIST_RATIO_THR 0.29


int main( int argc, char** argv )
{
  IplImage* img1, * img2;
  struct feature* feat1, * feat2, * feat;
  struct kd_node* kd_root = NULL;
  double d0, d1;
  int n1, n2, k, i, m = 0;
  int repeat;
  int n;
  
  if( argc != 4 )
    fatal_error( "usage: %s <img1> <dir> <number of repetitions>", argv[0] );

  img1 = cvLoadImage( argv[1], 1 );
  if( ! img1 )
    fatal_error( "unable to load image from %s", argv[1] );

  repeat = atoi(argv[3]);
  if (repeat < 0)
    fatal_error( "usage: %s <img1> <dir> <POSITIVE number of repetitions>", argv[0] );

  fprintf( stderr, "Finding features in %s...\n", argv[1] );
  n1 = _sift_features( img1, &feat1, 1/*SIFT_INTVLS*/, 2.2/*SIFT_SIGMA*/, 0.03/*SIFT_CONTR_THR*/, 10/*SIFT_CURV_THR*/, 0/*SIFT_IMG_DBL*/, 2/*SIFT_DESCR_WIDTH*/, 6/*SIFT_DESCR_HIST_BINS*/ );
  //n1 = _sift_features( img1, &feat1, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );
  fprintf( stderr, "%d features found\n", n1);

  img2 = cvLoadImage( argv[2], 1 );
  if( ! img2 )
    fatal_error( "unable to load image from %s", argv[2] );

  fprintf( stderr, "Finding features in %s...\n", argv[2] );
  n2 = _sift_features( img2, &feat2, 1/*SIFT_INTVLS*/, 2.2/*SIFT_SIGMA*/, 0.03/*SIFT_CONTR_THR*/, 10/*SIFT_CURV_THR*/, 0/*SIFT_IMG_DBL*/, 2/*SIFT_DESCR_WIDTH*/, 6/*SIFT_DESCR_HIST_BINS*/ );
  //n2 = _sift_features( img2, &feat2, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );
  fprintf( stderr, "%d features found\n", n2);

  kd_root = kdtree_build( feat2, n2 );

  for (n = 0; n < repeat; n++) {
	  m = 0;

	  for( i = 0; i < n1; i++ )
	  {
		  struct feature* nbrs[2];
		  feat = feat1 + i;
		  k = kdtree_bbf_knn( kd_root, feat, 2, nbrs, 5 );
		  if( k == 2 )
		  {
			  d0 = descr_dist_sq( feat, nbrs[0] );
			  d1 = descr_dist_sq( feat, nbrs[1] );
			  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			  {
				  m++;
			  }
		  }
	  }
  }

  cvReleaseImage( &img1 );
  cvReleaseImage( &img2 );
  kdtree_release( kd_root );
  free( feat1 );
  free( feat2 );
  return 0;
}
