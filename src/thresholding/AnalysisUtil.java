package thresholding;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class AnalysisUtil {

	public static ArrayList<Pair<Integer, Float>> getPeaks(float[] histData) {
		int count = 0;
		ArrayList<Pair<Integer, Float>> peakPoint = new ArrayList<>();
		// mathematical calculation of average values, variance and standard deviation
		// standard deviation and average are later used to look for peaks
		float average = 0;
		for (int k = 0; k < 256; k++) {
			average += histData[k];
		}
		average = average / 256;
		float variance = 0;
		for (int j = 0; j < 256; j++) {
			variance += ((histData[j] - average) * (histData[j] - average));
		}
		variance = Math.round((variance / 256));
		double standardDeviation = Math.sqrt(Math.abs(variance));
		double minPeakHeight;
		if((average*5) < (average + standardDeviation)) {
			minPeakHeight = average*5;
		} else {
			minPeakHeight = average + standardDeviation;
		}
		for (int l = 0; l < 256; l++) {
			if (histData[l] > (minPeakHeight) && l < 256) {
				if (peakPoint.size() == count) {
					peakPoint.add(new Pair<Integer, Float>(l, histData[l]));
				} else if (peakPoint.size() == (count + 1) && peakPoint.get(count).getFloat() < histData[l]) {
					peakPoint.set(count, new Pair<Integer, Float>(l, histData[l]));
				}
				if (l == 255 || (histData[l + 1] < (minPeakHeight))) {
					count++;
				}
			}
		}
		// testing, also shows how to get position and value of a pair
		// for(int p = 0; p<peakPoint.size(); p++) {
		// System.out.println("Pair " + p + ": position" + peakPoint.get(p).getInt() +
		// ", value " + peakPoint.get(p).getFloat());
		// }
		return peakPoint;
	}
	
	/**converts a mat to grayscale
	 * 
	 * @param src
	 * @param dest
	 */
	public static void convertMatToGray(Mat src, Mat dest) {
		switch(src.channels()) {
			case 4:Imgproc.cvtColor(src, dest, Imgproc.COLOR_RGBA2GRAY); break;
			case 3:Imgproc.cvtColor(src, dest, Imgproc.COLOR_RGB2GRAY); break;
			default: break;
		}
	}
	
	public static void showImage(Mat mat, String windowName, Boolean showOrSave) {
		if (true) {
			BufferedImage image = null;
			try {
				image = matToBufferedImage(mat, !showOrSave);
			} catch (Exception e) {
				e.printStackTrace();
			}
			if (!showOrSave) {
				JFrame frame = new JFrame();

				int rws = mat.rows();
				int cls = mat.cols();
				while (rws*cls > 1000000) {
	                cls/=2;
	                rws/=2;
	            }
				frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
				frame.setTitle(windowName);
				frame.setSize(cls + 60, rws + 40);
				frame.setLocation(50, 50);
				frame.setVisible(true);
				frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

				ImageIcon icon = new ImageIcon(image);
				JLabel label = new JLabel(icon);
				label.setBounds(0, 0, image.getWidth(null), image.getHeight(null));
				label.setVisible(true);
				frame.add(label);
			} else {
				try {
					File outputfile = new File(windowName + ".png");
					ImageIO.write(image, "png", outputfile);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	/**
	 * Converts a Matrix of an image to its corresponding BufferedImage
	 * 
     * @param mat input image
     * @param wantToCrop limits
	 * @return the resulting BufferedImage
	 */
	public static BufferedImage matToBufferedImage(Mat mat) throws Exception {
	    return matToBufferedImage(mat, false);
	}
	public static BufferedImage matToBufferedImage(Mat mat, Boolean wantToCrop) throws Exception {
		BufferedImage image;
		Mat matb = reverseChannels(mat);
		
        if (wantToCrop) {
            Size sz;
            while (matb.cols()*matb.rows() > 1000000) {
                sz = new Size(matb.cols()/2,matb.rows()/2);
                Imgproc.resize(matb, matb, sz);
            }
        }

		int type;

		if (matb.channels() == 1) {
			type = BufferedImage.TYPE_BYTE_GRAY;
		} else if (matb.channels() == 3) {
			type = BufferedImage.TYPE_3BYTE_BGR;
		} else if (matb.channels() == 4) {
			type = BufferedImage.TYPE_4BYTE_ABGR;
		} else {
			throw new Exception("undefined Channel Count");
		}

		int dataLength = matb.channels() * matb.cols() * matb.rows();
		byte[] buffer = new byte[dataLength];
		image = new BufferedImage(matb.cols(), matb.rows(), type);
		matb.get(0, 0, buffer);

		final byte[] bimPixels = (((DataBufferByte) image.getRaster().getDataBuffer()).getData());
		System.arraycopy(buffer, 0, bimPixels, 0, buffer.length);

		return image;
	}
	
	/**
	 * Reverses the order of channels with each use abgr <-> rgba bgr <-> rgb The
	 * intention is to have the alpha channel at its correct spot
	 * 
	 * @param mat input image
	 * @return the reordered mat
	 */
	static Mat reverseChannels(Mat mat) {
		Mat m = mat.clone();
		ArrayList<Mat> preSplittedImage = new ArrayList<>();
		Core.split(m, preSplittedImage);
		ArrayList<Mat> reordered = new ArrayList<>();
		if (m.channels() == 3) {
			reordered.add(preSplittedImage.get(2));
			reordered.add(preSplittedImage.get(1));
			reordered.add(preSplittedImage.get(0));
		} else if (m.channels() == 4) {
			reordered.add(preSplittedImage.get(3));
			reordered.add(preSplittedImage.get(2));
			reordered.add(preSplittedImage.get(1));
			reordered.add(preSplittedImage.get(0));
		}
		/*
		 * else if (m.channels() == 2) { //test that may help for 2 channel pictures
		 * (currently untested) reordered.add(preSplittedImage.get(1));
		 * reordered.add(preSplittedImage.get(1));
		 * reordered.add(preSplittedImage.get(1));
		 * reordered.add(preSplittedImage.get(0)); }
		 */
		else {
			reordered = preSplittedImage;
		}
		Core.merge(reordered, m);
		return m;
	}
	
	/**
     * Converts a BufferedImage to its Matrix representation.
     * 
     * @param bi input image
     * @return the resulting byte matrix
     */
    public static Mat bufferedImageToMat(BufferedImage bi) {

        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC4);

        byte[] matData = new byte[4 * bi.getWidth() * bi.getHeight()];
        for(int x = 0; x < bi.getWidth(); x++) {
            for (int y = 0; y < bi.getHeight(); y++) {
                int color = bi.getRGB(x, y);
                int i = (y * bi.getWidth() + x) * 4;
                matData[i + 2] = (byte) (color & 0xff);
                matData[i + 1] = (byte) ((color >> 8) & 0xff);
                matData[i] = (byte) ((color >> 16) & 0xff);
                matData[i + 3] = (byte) ((color >> 24) & 0xff);
            }
        }
        mat.put(0, 0, matData);

        showImage(mat, "bufferedImageToMat", true);

        return mat;
    }

}
