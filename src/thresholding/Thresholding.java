package thresholding;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

//class Pair to safe pairs of values with getX for each value
class Pair<I, F> {
	public final I i;
	public final F f;

	Pair(I i, F f) {
		this.i = i;
		this.f = f;
	}

	public I getInt() {
		return i;
	}

	public F getFloat() {
		return f;
	}

}

public class Thresholding {
	
	static boolean treshDebug = true;

	public static void main(String[] args) { 
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//Load sample image to demonstrate otsu-Threshold
		BufferedImage img = null;
		try {
			img = ImageIO.read(new File("src/resources/sampleImage1.png"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		};
		Mat sampleMat = AnalysisUtil.bufferedImageToMat(img);
		//AnalysisUtil.showImage(sampleMat, "Image pre edit", false);
		Mat finishedMat = threshold(sampleMat);
		//AnalysisUtil.showImage(finishedMat, "Image post edit", false);
	}
	
	/**
	 * Applies thresholding algorithms to the image
	 * 
	 * @param thresholding will be applied to this inital mat
	 * @return the thresholded image
	 */
	public static Mat threshold(Mat mat) {
//		Turn image to grayscale
		Mat treshImg = new Mat(mat.size(), CvType.CV_8UC1);
		Mat grayImg = new Mat(treshImg.rows(), treshImg.cols(), CvType.CV_8UC1);
		AnalysisUtil.convertMatToGray(mat, grayImg);
		if (grayImg.channels() != 1) {
			throw new IllegalArgumentException("Please pass a grayscale Image");
		}

//		actual thresholding
		Mat histogram = histogram(grayImg);
		Mat thresholdedMat = new Mat();
		Boolean backIsBlack = backgroundCheck(grayImg, histogram);
		if (getNeededThresholdCount(histogram, grayImg) == 1) { // 1 Threshold -> OpenCV Tresholding
			System.out.println("OpenCV");
			if (backIsBlack) {
				Imgproc.threshold(grayImg, thresholdedMat, 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY);
			} else {
				Imgproc.threshold(grayImg, thresholdedMat, 0, 255, Imgproc.THRESH_OTSU + Imgproc.THRESH_BINARY_INV);
			}
		} else if (getNeededThresholdCount(histogram, grayImg) > 1) { // 2 Thresholds -> custom otsu Tresholding
			System.out.println("custom otsu");
			thresholdedMat = otsu2(grayImg, histogram);
		} else {
			return mat;
		}
		
		// Debug
		if (treshDebug) {
			AnalysisUtil.showImage(grayImg, "input img", false);
			AnalysisUtil.showImage(thresholdedMat, "otsu modified", false);
			printHist(histogram, "histogram with otsu");
		}
		
		return thresholdedMat;
	}
	
	/**The application of otsu onto an image
	 * 
	 * @param mat
	 * @param hist
	 * @return 
	 */
	private static Mat otsu2(Mat mat, Mat hist) {
		Mat treshImg = new Mat(mat.size(), CvType.CV_8UC1);
		double[] otsu = otsu2Calc(hist);
		Boolean backgroundInOtsu = backgroundInOtsu(otsu, getBackgroundPeaks(mat));
		for (int x = 0; x < mat.rows(); x++) {
			for (int y = 0; y < mat.cols(); y++) {
				double[] pixel = mat.get(x, y);
				if (pixel[0] <= otsu[0] || pixel[0] >= otsu[1]) {
					if (!backgroundInOtsu) { 
						treshImg.put(x, y, 0); // Black for outside
					} else {
						treshImg.put(x, y, 255); // White for outside (img is already inverse)
					}
				} else if (pixel[0] > otsu[0] && pixel[0] < otsu[1]) {
					if (!backgroundInOtsu) {
						treshImg.put(x, y, 255); // White for inside
					} else {
						treshImg.put(x, y, 0); // Black for inside (img is already inverse)
					}
				}
			}
		}
		return treshImg;
	}
	
	/**Only returns True if ALL backgroundPeaks are between otsu's 2 values
	 * 
	 * @param otsu
	 * @param peaks
	 * @return
	 */
	private static Boolean backgroundInOtsu (double[] otsu, ArrayList<Pair<Integer, Float>> peaks) {	
		for (int i = 0; i < peaks.size(); i++) {
			if (peaks.get(i).getInt() <= otsu[0] || peaks.get(i).getInt() >= otsu[1]) {
				return false;
			}
		}	
		return true;
	}
	
	/**
	 * Calculates 2 thresholds with the Otsu-Algortihm
	 * 
	 * @source https://stackoverflow.com/questions/22706742/multi-otsumulti-thresholding-with-opencv
	 * @param histogram
	 * @return 2 thresholds
	 */
	private static double[] otsu2Calc(Mat histogram) {
		double[] thresholds = new double[2];
		int[] hist = new int[256];
		double[] histogramPoint;

		// total number of pixels
		int n = 0;

		// accumulate image histogram and total number of pixels
		for (int i = 0; i < histogram.rows(); i++) {
			histogramPoint = histogram.get(i, 0);
			hist[i] = (int) histogramPoint[0];
			for (int j = 0; j <= hist[i]; j++) {
				n++;
			}
		}

		double w0k = 0;
		double w1k = 0;
		double w2k;
		double m0;
		double m1;
		double m2;
		double currVarB;
		double maxBetweenVar;
		double m0k = 0;
		double m1k = 0;
		double m2k;
		double mt = 0;

		maxBetweenVar = 0;
		for (int k = 0; k <= 255; k++) {
			mt += k * (hist[k] / (double) n);
		}

		for (int t1 = 0; t1 <= 255; t1++) {
			w0k += hist[t1] / (double) n; // Pi
			m0k += t1 * (hist[t1] / (double) n); // i * Pi
			m0 = m0k / w0k; // (i * Pi)/Pi

			w1k = 0;
			m1k = 0;

			for (int t2 = t1 + 1; t2 <= 255; t2++) {
				w1k += hist[t2] / (double) n; // Pi
				m1k += t2 * (hist[t2] / (double) n); // i * Pi
				m1 = m1k / w1k; // (i * Pi)/Pi

				w2k = 1 - (w0k + w1k);
				m2k = mt - (m0k + m1k);

				if (w2k <= 0)
					break;

				m2 = m2k / w2k;

				currVarB = w0k * (m0 - mt) * (m0 - mt) + w1k * (m1 - mt) * (m1 - mt) + w2k * (m2 - mt) * (m2 - mt);

				if (maxBetweenVar < currVarB) {
					maxBetweenVar = currVarB;
					thresholds[0] = t1;
					thresholds[1] = t2;
				}
			}
		}
		return thresholds;
	}
	
	/**
	 * Compares histograms of background and image in order to determine how otsu
	 * should recolor the image
	 * 
	 * @param mat  The mat of the image
	 * @param hist The histogram of the image
	 * @return isBlack
	 */
	private static Boolean backgroundCheck(Mat mat, Mat hist) {
//		Create peakPoints
		ArrayList<Pair<Integer, Float>> peakPoints = getPeaks(hist);
		ArrayList<Pair<Integer, Float>> peakPointsBackground = getBackgroundPeaks(mat);

//		compare peaks via position
		int foundPeaks = 0;
		int averageValue = 0;
		for (int i = 0; i < peakPointsBackground.size(); i++) {
			for (int j = 0; j < peakPoints.size(); j++) {
				if (peakPoints.get(j).getInt().equals(peakPointsBackground.get(i).getInt())) {
					foundPeaks++;
				}
			}
			averageValue += peakPointsBackground.get(i).getInt();
		}
		averageValue = averageValue / peakPointsBackground.size();
		if (foundPeaks == peakPointsBackground.size()) {
			return averageValue < 137;
		} else {
			return true;
		}		
	}
	
	/**
	 * Creates a Histogram of a given Image with one channel.
	 * 
	 * @param img the image one wants to convert to a histogram as a one channel double mat
	 * @return one column vector
	 */
	public static Mat histogram(Mat grayImg) {
		Mat hist = new Mat();
		MatOfInt histSize = new MatOfInt(256);
		List<Mat> source = new ArrayList<>();
		source.add(grayImg);
		Imgproc.calcHist(source, new MatOfInt(0), new Mat(), hist, histSize, new MatOfFloat(0f, 256f));
		Core.normalize(hist, hist, 0, hist.rows(), Core.NORM_MINMAX, -1, new Mat()); // normalize for equal distribution
		return hist;
	}
	
	/**
	 * Will print the histogram of the image into a window (while debug is enabled)
	 * 
	 * @param hist calculated histogram of image
	 */
	public static void printHist(Mat hist, String winName) {
//		Colors
		Scalar backColor = new Scalar(255, 255, 255, 255);
		Scalar otsuColor = new Scalar(0, 0, 255, 255);
		Scalar peakColor = new Scalar(255, 0, 0, 255);
		Scalar lineColor = new Scalar(0, 0, 0, 255);
		Mat histImg = new Mat(256, 300, CvType.CV_8UC3, backColor);
		int binW = 500 / 256;
		Core.normalize(hist, hist, 0, histImg.rows(), Core.NORM_MINMAX, -1, new Mat());

//		<-- Print histogram --> 
		for (int i = 0; i < 256; i++) {
			double[] dummy = hist.get(i, 0);
			Point pt1 = new Point(binW * (double) i, 1);
			Point pt2 = new Point(binW * (double) i, (int) Math.round((dummy[0])));
			Imgproc.line(histImg, pt1, pt2, lineColor, 3, 8, 0);
		}
		Core.flip(histImg, histImg, 0);

//		<-- Print otsu lines -->
		double[] otsu = Thresholding.otsu2Calc(hist);
		for (int x = 0; x < otsu.length; x++) {
			Point otsuPoint1 = new Point((int) otsu[x], 1);
			Point otsuPoint2 = new Point((int) otsu[x], 300);
			Imgproc.line(histImg, otsuPoint1, otsuPoint2, otsuColor, 2, 4, 0);
		}
		
//		<-- Print peaks -->
		ArrayList<Pair<Integer, Float>> peakPoints = getPeaks(hist);
		for (int p = 0; p < peakPoints.size(); p++) {
			Point peakPoint1 = new Point((int) peakPoints.get(p).getInt(), 1);
			Point peakPoint2 = new Point((int) peakPoints.get(p).getInt(), 300);
			Imgproc.line(histImg, peakPoint1, peakPoint2, peakColor, 1, 4, 0);
		}

		// Print
		AnalysisUtil.showImage(histImg, winName, false);
	}


	/**Creates an array filled with 0 = image Peak and 1 = Background Peak
	 * 
	 * @param histogram
	 * @param mat
	 * @return the array
	 */
	private static int[] createPeakArray(Mat mat, Mat histogram) {
		ArrayList<Pair<Integer, Float>> peakPoints = getPeaks(histogram);
		ArrayList<Pair<Integer, Float>> peakPointsBackground = getBackgroundPeaks(mat);		
		int[] peaks = new int[peakPoints.size()+1];
		int i = 0;
		while (i < peakPoints.size()) {
			int j = 0;
			while (j < peakPointsBackground.size()) {
				if (peakPoints.get(i).getInt().equals(peakPointsBackground.get(j).getInt())) {
					peaks[i] = 1;
					j = peakPointsBackground.size();
				} else {
					peaks[i] = 0;
					j++;
				}
			}
			i++;
		}
		peaks[peakPoints.size()] = 2;
		return peaks;
	}
	
	/**Calculates the needed Threshold Count
	 * 
	 * @param histogram
	 * @param mat
	 * @return 
	 */
	static int getNeededThresholdCount(Mat histogram, Mat mat) {
		ArrayList<Pair<Integer, Float>> peakPoints = getPeaks(histogram);
		if (peakPoints.size() <= 2 ) {
			return 1;
		}
		int[] peaks = createPeakArray(mat, histogram);
		int threshCount = 0;
		int k = 0;
		while ( k < peakPoints.size() ) {
			if (peaks[k] == 0) {
				k++;
			} else {
				int j = k;
				while (peaks[j] == 1) {
					j++;
				}
				if (k > 0) {
					if (peaks[k-1] == 0 && peaks[j] == 0) { 
						return 2;
					}
				}
				threshCount++;
				k = j;
			}
		}
		return threshCount;
	}
	
	/**Creates the List of Peaks, needed because AnalysisUtil.getPeaks requires float[]
	 * 
	 * @param hist
	 * @return
	 */
	private static ArrayList<Pair<Integer, Float>> getPeaks(Mat hist) {
		float[] floatHist = new float[(int) hist.total()];
		hist.get(0, 0, floatHist);
		return AnalysisUtil.getPeaks(floatHist);
	}
	
	/**Creates a masked image and returns the Peaks of that masked image
	 * 
	 * @param mat
	 * @return the peaks of the border of the image
	 */
	private static ArrayList<Pair<Integer, Float>> getBackgroundPeaks(Mat mat) {
		Mat histMask = new Mat();
		MatOfInt histSize = new MatOfInt(256);
		List<Mat> source = new ArrayList<>();
		source.add(mat);
		Point pt1 = new Point(0, 0);
		Point pt2 = new Point(mat.cols(), mat.rows());
		Point pt3 = new Point(3, 3);
		Point pt4 = new Point(mat.cols() - 4d, mat.rows() - 4d);
		Scalar black = new Scalar(0, 0, 0);
		Scalar white = new Scalar(255, 255, 255);
		Mat maskMat = new Mat(mat.rows(), mat.cols(), CvType.CV_8UC1);
		Imgproc.rectangle(maskMat, pt1, pt2, white, -1); // draw background
		Imgproc.rectangle(maskMat, pt3, pt4, black, -1); // draw non area
		Imgproc.calcHist(source, new MatOfInt(0), maskMat, histMask, histSize, new MatOfFloat(0f, 256f));
		Core.normalize(histMask, histMask, 0, histMask.rows(), Core.NORM_MINMAX, -1, new Mat()); // normalize for equal
		return getPeaks(histMask);
	}

}