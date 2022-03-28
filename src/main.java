package DetectImage;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

public class main {
	
	public static Vector<String> classNames() throws IOException  {
				String f1 = "coco.names";
				Vector<String> class_names = new Vector<>();
				File file = new File(f1);
				FileReader fr = new FileReader(file);
				BufferedReader br = new BufferedReader(fr);
				String line;
				while((line=br.readLine())!= null) {
					class_names.add(line);
				}
		
		return class_names;
		
	}

	public static void main(String[] args) throws IOException {
	    
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		
		Vector<String> class_names = classNames();
		System.out.println(class_names);
		
		
		
	  //load yolo network from darknet  (a neural network framework)
		String f2 = "yolov3.weights";
		String f3 = "yolov3.cfg";
		
		Net net = Dnn.readNetFromDarknet(f3, f2);
		
		if(net.empty()) {
			System.out.println("Reading Net Error");
		}
		// layers
		List<String> names = new ArrayList<>();
		List<Integer> outLayers = net.getUnconnectedOutLayers().toList(); //3
		System.out.println(outLayers);
		
		List<String> layersNames = net.getLayerNames();//254

		outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
		
		//read the img
		String img = "per1.jpg";
		Mat im = Imgcodecs.imread(img);
		if(im.empty()) {
			System.out.println("Reading Image error");
		}
		
		//copy it to a frame
		Mat frame = new Mat();
		Size s1 = new Size(im.cols(),im.rows());
		Imgproc.resize(im,frame, s1);
		
		//resize the img  to 416x416
		Mat resized = new Mat();
		Size s2 = new Size(416, 416);
		Imgproc.resize(im, resized, s2);
		
		float scale = 1.0F/255.0F; 
		
		Mat inputBlob = Dnn.blobFromImage(resized,scale,s2,new Scalar(0),true,false);
		
		net.setInput(inputBlob, "data");
		
		List<Mat> detectionMat = new ArrayList<Mat>();
		net.forward(detectionMat,names);
		
		float conf = 0.6f;
		List<Integer> clsIds = new ArrayList<>();
		List<Float> confs = new ArrayList<>();
		List<Rect> rects = new ArrayList<>();
		
		
		for(Mat detect : detectionMat) { //size = 3 or for(int i = 0;i<detectionMat.size();i++)
			for(int j = 0;j<detect.rows();j++) {
				Mat row = detect.row(j); //select data per data 1*85
				Mat scores = row.colRange(5, detect.cols()); //prob dyal kol object 1*80
				Core.MinMaxLocResult m = Core.minMaxLoc(scores);
				float confidence = (float)m.maxVal;
				Point classIdPoint = m.maxLoc;
				if(confidence > conf) {
					int centerx = (int)(row.get(0, 0)[0]*im.cols());
					int centery = (int)(row.get(0, 1)[0]*im.rows());
					int width = (int)(row.get(0, 2)[0]*im.cols());
					int height = (int)(row.get(0, 3)[0]*im.rows());
					
					int left = centerx - width/2;
					int top = centery -height/2;
					
					clsIds.add((int)classIdPoint.x); //puisque y = 0 (1 row) label id of the object = classIdPoint.x 
					confs.add((float)confidence);
					rects.add(new Rect(left,top,width,height));
					
					
				}
			}
		}
		
		for(int i = 0;i<clsIds.size();i++) {
			System.out.println("class : "+class_names.get(clsIds.get(i)) + " with probability = "+ confs.get(i)*100 + " %");
		}
		
		float nmsthreshold = 0.5f;
		MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
		Rect[] boxesArray = rects.toArray(new Rect[0]);
		MatOfInt indices = new MatOfInt();
		MatOfRect boxes = new MatOfRect(boxesArray);
		Dnn.NMSBoxes(boxes, confidences, conf, nmsthreshold, indices);
		
		int[] ind = indices.toArray();
		for(int i = 0 ; i<ind.length;i++) {
			int idx = ind[i];
			
			if(ind[i] == 0) {
			Rect box = boxesArray[idx];
			Imgproc.rectangle(frame,box.tl(),box.br(),new Scalar(0,0,255),2);
			Imgproc.putText(frame,class_names.get(clsIds.get(i)),box.tl(),1,0.75,new Scalar(255,0,0));
			System.out.println(box);
			}
			
		}
		
		HighGui gui = new HighGui();
		gui.namedWindow("new window");
		
		gui.imshow("new window", frame);
		
		int k;
		k= gui.waitKey(0);
		
		if(k==81) {
			gui.waitKey(0);
			gui.destroyAllWindows();
		}
		
       System.out.println("end congra !!!");
	}

}
