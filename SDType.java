package classifiers;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;


public class SDType implements MultiLabelLearner, Serializable{
	 
	 protected int numInstances = 0;
	 protected Map<Integer,Double> propNumInstances;
	 
	 protected Map<Integer,Double> typeDistribution;
	 protected Map<Integer,Map<Integer,Double>> propTypesDistribution;
	 
	 protected Map<Integer,Double> propWeights;
	 
	 protected Set<Attribute> features = new HashSet<Attribute>();
	 protected Set<Attribute> labels = new HashSet<Attribute>();
	 
	 protected double threshold = Double.NaN;
	 
	 protected Map<Integer,Integer> labelIndices = new HashMap<Integer,Integer>();	
	 
	 public ArrayList<Double> precision = new ArrayList<Double>();
	 public ArrayList<Double> recall = new ArrayList<Double>();
	 public ArrayList<Double> confidence = new ArrayList<Double>();

	 public SDType() throws IOException {
		 this.typeDistribution = new HashMap<Integer,Double>();
		 this.propNumInstances = new HashMap<Integer,Double>();
		 this.propTypesDistribution = new HashMap<Integer,Map<Integer,Double>>();
		 this.propWeights = new HashMap<Integer,Double>();
	 }
	 
	 public SDType(double threshold) throws IOException {
		 this();
		 this.threshold = threshold; 
	 }
	 
	 
	 public MultiLabelOutput makePrediction(Instance instance) throws Exception,
			InvalidDataException, ModelInitializationException {
		 
		double[] confidences = new double[labelIndices.size()];
		Arrays.fill(confidences, 0.0);

		Collection<Integer> feats = new ArrayList<Integer>();
		for (Attribute feat: features){
			 int featID = feat.index();
			 if (instance.value(featID)==1)
				 feats.add(featID);				 
		}
		 
		if (feats!=null) {
			for (int classID: labelIndices.keySet()) {
				 double conf = computeSDTypeConfidence(instance, classID, feats);
				 confidences[labelIndices.get(classID)] = conf;
			}
		}
		
		if (Double.isNaN(threshold))
			return new MultiLabelOutput(confidences);
		else
			return new MultiLabelOutput(confidences, threshold); 
	 }
	 
	 
	public void build(MultiLabelInstances data) throws Exception,
			InvalidDataException {
			 		
		labels = data.getLabelAttributes();
		features = data.getFeatureAttributes();
		
		for (int i=0; i<data.getLabelIndices().length; i++) 
			labelIndices.put(data.getLabelIndices()[i], i);
		 
		 // Calculate type distributions
		for (int i=0; i<data.getNumInstances(); i++) {
			Instance instance = data.getDataSet().instance(i);
			 	

			numInstances++;
			 
			Collection<Integer> feats = new ArrayList<Integer>();
			for (Attribute feat: features) {
				int featID = feat.index();
				 if (instance.value(featID)==1)
					 feats.add(featID);	
			}

			Collection<Integer> types = new ArrayList<Integer>();
			for (int type: labelIndices.keySet()) {
				 if (instance.value(type)==1)
					 types.add(type);		
			}
			

			 for (int type: types) {
				 
				 if (!typeDistribution.containsKey(type))
					 typeDistribution.put(type, 0.0);
				 typeDistribution.put(type, typeDistribution.get(type)+1.0);
				 
				 for (int feat: feats) {
					 if (!propTypesDistribution.containsKey(feat))
						 propTypesDistribution.put(feat, new HashMap<Integer,Double>());
					 HashMap<Integer,Double> typeDistributionGivenProp = (HashMap<Integer, Double>) propTypesDistribution.get(feat);
					 if (!typeDistributionGivenProp.containsKey(type))
						 typeDistributionGivenProp.put(type, 0.0);
					 typeDistributionGivenProp.put(type, typeDistributionGivenProp.get(type)+1.0); 
					 
					 if (!propNumInstances.containsKey(feat))
						 propNumInstances.put(feat, 0.0);
					 propNumInstances.put(feat, propNumInstances.get(feat)+1.0); 
				 }
			 }
			 
		 }
		 

		 
		 // Normalize distributions
		 for (int type: typeDistribution.keySet()) 
			 typeDistribution.put(type, typeDistribution.get(type)/(double)numInstances);

		 for (int prop: propTypesDistribution.keySet()) {
			 HashMap<Integer,Double> typeDist = (HashMap<Integer, Double>) propTypesDistribution.get(prop);
			 for (int type: typeDist.keySet()) 
				 typeDist.put(type, typeDist.get(type)/propNumInstances.get(prop));
		 }

		 
		 // Calculate weights
		 for (int prop: propTypesDistribution.keySet()) 
			 propWeights.put(prop, 0.0);


		 for (int type: typeDistribution.keySet()) {
			 double mean = typeDistribution.get(type);
			 
			 for (int prop: propWeights.keySet()) { 
				 double var;
				 if (propTypesDistribution.get(prop).containsKey(type))
					  var = Math.pow(propTypesDistribution.get(prop).get(type) - mean, 2);
				 else
					 var = Math.pow(mean, 2);
				 propWeights.put(prop, propWeights.get(prop)+var);
			 }
		 }
		 
		 // Find threshold which optimizes F-measure
		 if (Double.isNaN(this.threshold))
			 this.threshold = findThreshold(data,this,1);

	 }
	 
	 private double computeSDTypeConfidence(Instance instance, int classID, Collection<Integer> props) {
		 double conf = 0;
		 double normalizationFactor = 0;
		 
		 for (int prop: props) { 
			 if (propTypesDistribution.containsKey(prop))
				 if (propTypesDistribution.get(prop).containsKey(classID)) {
					 conf += propWeights.get(prop) * propTypesDistribution.get(prop).get(classID);			
					 normalizationFactor += propWeights.get(prop); 
				 }
		 }
		 if (normalizationFactor==0)
			 return 0;
		 
		 normalizationFactor = 1/normalizationFactor; 

		 conf *= normalizationFactor;

		 return conf;
	 }
	 
	public boolean isUpdatable() {
		return false;
	}

	public MultiLabelLearner makeCopy() throws Exception {
		return new SDType();
	}

	public void setDebug(boolean arg0) {
		// TODO Auto-generated method stub
		
	}
	
	public static double findThreshold(MultiLabelInstances data, MultiLabelLearner classifier, double beta) throws InvalidDataException, ModelInitializationException, Exception {
		 
		 TreeMap<Double,Integer> tp = new TreeMap<Double,Integer>();
		 TreeMap<Double,Integer> fp = new TreeMap<Double,Integer>();
		 
		 double numPos = 0;
		 
		 for (int i=0; i<data.getNumInstances(); i++) {
			Instance instance = data.getDataSet().instance(i);
			MultiLabelOutput prediction = classifier.makePrediction(instance);

			for (int classID = 0; classID<data.getLabelIndices().length; classID++) {
				double conf = - prediction.getConfidences()[classID];
				
				if (!tp.containsKey(conf)) tp.put(conf, 0);
				if (!fp.containsKey(conf)) fp.put(conf, 0);
				
				if (instance.value(data.getLabelIndices()[classID])==1.0) {
					tp.put(conf, tp.get(conf)+1);
					numPos++;
					
				} else { 
					fp.put(conf, fp.get(conf)+1);
				}
			}
			
		 }	
		 
		 double currFMeasure;
		 double maxFMeasure = 0;
		 double argmaxConf = 0;
		 double currPrecision;
		 double currRecall;
		 double currTP = 0;
		 double currFP = 0;
		 
		 ArrayList<Double> precision = new ArrayList<Double>();
		 ArrayList<Double> recall = new ArrayList<Double>();
		 
		 for (Iterator<Double> iter=tp.descendingKeySet().descendingIterator(); iter.hasNext();) {
			 double conf = iter.next();
			 
			 if (!Double.isNaN(conf)) {
				 
				 currTP += tp.get(conf);
				 currFP += fp.get(conf);
				 
				 currPrecision = currTP/(currTP+currFP);
				 currRecall = currTP/numPos;
				 
				 if (recall.isEmpty() || currRecall>recall.get(recall.size()-1)+0.01) {
					 precision.add(currPrecision);
					 recall.add(currRecall);
				 }
				 
				 currFMeasure = (1+beta*beta)*(currPrecision*currRecall)/(beta*beta*currPrecision+currRecall);
				 
				 if (currFMeasure>=maxFMeasure) {
					 maxFMeasure = currFMeasure;
					 argmaxConf = conf;
				 }
			 }
		 }
	
		 double threshold = -argmaxConf;
		 
		 return threshold;
	}	


}
