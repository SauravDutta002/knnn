package knn;

import java.io.*;
import java.util.*;

public class kNearestNeighbour {

	public static List<double[]> readCSV(String filePath) throws IOException {
		List<double[]> dataList = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(filePath));
		String line;
		while ((line = reader.readLine()) != null) {
			String[] values = line.split(",");
			double[] row = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
			dataList.add(row);
		}
		reader.close();
		return dataList;
	}

	public static double getEuclideanDistance(double[] a, double[] b) {
		double sum = 0.0;
		for (int i = 0; i < a.length - 1; i++) {
			sum += Math.pow(a[i] - b[i], 2);
		}
		return Math.sqrt(sum);
	}

	public static double getManhattanDistance(double[] a, double[] b) {
		double sum = 0.0;
		for (int i = 0; i < a.length - 1; i++) {
			sum += Math.abs(a[i] - b[i]);
		}
		return sum;
	}

	public static int guessLabelKNN(List<double[]> trainData, double[] testPoint, int k, String distanceMetric) {
		List<Double> distances = new ArrayList<>();
		for (double[] trainPoint : trainData) {
			double distance = (distanceMetric.equals("Euclidean")) ? getEuclideanDistance(trainPoint, testPoint)
					: getManhattanDistance(trainPoint, testPoint);
			distances.add(distance);
		}

		List<Integer> nearestLabels = new ArrayList<>();
		for (int i = 0; i < k; i++) {
			int index = distances.indexOf(Collections.min(distances));
			nearestLabels.add((int) trainData.get(index)[trainData.get(index).length - 1]);
			distances.set(index, Double.MAX_VALUE);
		}

		return mostFrequent(nearestLabels);
	}

	public static int mostFrequent(List<Integer> list) {
		Map<Integer, Integer> frequencyMap = new HashMap<>();
		for (int i : list) {
			frequencyMap.put(i, frequencyMap.getOrDefault(i, 0) + 1);
		}

		int maxFrequency = -1;
		int mostFrequentLabel = -1;
		for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
			if (entry.getValue() > maxFrequency) {
				maxFrequency = entry.getValue();
				mostFrequentLabel = entry.getKey();
			}
		}
		return mostFrequentLabel;
	}

	public static double calculatingAccuracyOfModel(List<double[]> trainData, List<double[]> testData, int k,
			String distanceMetric) {
		int correctPredictions = 0;
		for (double[] testPoint : testData) {
			int predicted = guessLabelKNN(trainData, testPoint, k, distanceMetric);
			int actual = (int) testPoint[testPoint.length - 1];
			if (predicted == actual) {
				correctPredictions++;
			}
		}
		return (double) correctPredictions / testData.size() * 100;
	}

	public static void main(String[] args) throws IOException {
		List<double[]> group1 = readCSV("data/dataSet1.csv");
		List<double[]> group2 = readCSV("data/dataSet2.csv");

		int k = 2;
		String distanceMetric = "Euclidean";

		double accuracy1 = calculatingAccuracyOfModel(group1, group2, k, distanceMetric);
		double accuracy2 = calculatingAccuracyOfModel(group2, group1, k, distanceMetric);

		System.out.println("Accuracy of model when Train on group1, Test on group2 using " + distanceMetric
				+ " distance: " + accuracy1 + "%");
		System.out.println("Accuracy of model when Train on group2, Test on group1 using " + distanceMetric
				+ " distance: " + accuracy2 + "%");
		System.out.println(
				"Average Accuracy using " + distanceMetric + " distance: " + (accuracy1 + accuracy2) / 2 + "%");
	}
}
