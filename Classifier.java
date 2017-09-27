/** J2ME-Bayes
 *
 * Rafael Alencar  rafjaa at gmail.com
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * 
 * J2ME-Bayes is a general purpose Naive Bayes Classifier to the Java
 * Micro Edition platform. It consists in a simple class, easy to use
 * for many proposals, including:
 *    - Classify SMS spam;
 *    - Classify SMS by author, category, etc;
 *    - Detect the language of a text;
 *    - Classify any kind of document.
 *
 * This algorithm is inspired in the book "Programming Collective Intelligence",
 * by Toby Segaran.
 *
 * @author Rafael Alencar
 * @version 0.1 November/2010
 */

import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

public class Classifier {

    private Hashtable featureCounter,
            categoryCounter;
    
    public Classifier() {
        featureCounter = new Hashtable();
        categoryCounter = new Hashtable();
    }

    /**
     * Implement this method if you need load a previous classifier train
     * to the hashtables above.
     */
    public void load(){
        throw new RuntimeException("Not implemented yet.");
    }

    /**
     * Implement this method if you need save the classifier train.
     */
    public void save(){
        throw new RuntimeException("Not implemented yet.");
    }

    /** A very simple string tokenizer.
     *
     * The J2ME plataform does not have:
     *     The StringTokenizer Class;
     *     The String split method;
     *     A RegExp engine.
     *
     * @param The string which will be parsed.
     * @return A Vector with the String tokens.
     */ 
    private Vector simpleTokenizer(String string){
        int ascii, space, start = 0;
        String stringTrim, parsedString = "";
        Vector tokens = new Vector();

        stringTrim = string.trim().toLowerCase();
        for(int i = 0; i < stringTrim.length(); i += 1){
            ascii = (int)stringTrim.charAt(i);
            if(ascii == 32 || (ascii > 47 && ascii < 58) || (ascii > 96 && ascii < 123))
                parsedString += stringTrim.charAt(i);
        }

        while(true){
            space = parsedString.indexOf(' ');
            if(space == -1){
                tokens.addElement(parsedString);
                break;
            }
            tokens.addElement(parsedString.substring(0, space));
            parsedString = parsedString.substring(space + 1).trim();
            start = space + 1;
        }
        return tokens;
    }

    private void incrementFeature(String feature, String category){
        Hashtable featureData = (Hashtable)featureCounter.get(feature);
        if(featureData == null){
            Hashtable newCount = new Hashtable();
            newCount.put(category, new Integer(1));
            featureCounter.put(feature, newCount);
            return;
        }
        Integer count = (Integer)featureData.get(category);
        int value = (count == null) ? 1 : count.intValue() + 1;
        featureData.put(category, new Integer(value));
        featureCounter.put(feature, featureData);
    }

    private void incrementCategory(String category){
        Integer count = (Integer)categoryCounter.get(category);
        int value = (count == null) ? 1 : count.intValue() + 1;
        categoryCounter.put(category, new Integer(value));
    }

    private int featureCount(String feature, String category){
        Hashtable featureData = (Hashtable)featureCounter.get(feature);
        if(featureData == null)
            return 0;
        Integer count = (Integer)featureData.get(category);
        return (count != null) ? count.intValue() : 0;
    }

    private int categoryCount(String category){
        Integer count = (Integer)categoryCounter.get(category);
        return (count != null) ? count.intValue() : 0;
    }

    private void trainer(Vector features, String category){
        String feature;
        while(!features.isEmpty()){
            feature = (String)features.firstElement();
            incrementFeature(feature, category);
            features.removeElementAt(0);
        }
        incrementCategory(category);
    }

    /**
    * Train the classifier parsing the item with the simpleTokenizer.
    */
    public void train(String item, String category){
        Vector features = simpleTokenizer(item);
        trainer(features, category);
    }

    /**
    * Train the classifier using an external parser to get the features.
    */
    public void train(Vector features, String category){
        trainer(features, category);
    }

    private int featureOccurrence(String feature){
        Hashtable featureData = (Hashtable)featureCounter.get(feature);
        if(featureData == null)
            return 0;
        int count = 0;
        Enumeration keys = featureData.keys();
        while(keys.hasMoreElements()){
            Integer val = (Integer)featureData.get(keys.nextElement());
            count += val.intValue();
        }
        return count;
    }

    private double featureProbability(String feature, String category){
        int count = categoryCount(category);
        return (count == 0) ? 0.0 : (double)featureCount(feature, category) / count;
    }

    private double weightedProbability(String feature, String category){
        final double ASSUMED_PROBABILITY = 0.5;
        final double WEIGHT = 1.0;
        double basicProbability = featureProbability(feature, category);
        int occurrence = featureOccurrence(feature);
        return (double)(WEIGHT * ASSUMED_PROBABILITY + occurrence * basicProbability) / (WEIGHT + occurrence);
    }

    private double documentProbability(String item, String category){
        Vector features = simpleTokenizer(item);
        String feature;
        double probability = 1;
        while(!features.isEmpty()){
            feature = (String)features.firstElement();
            features.removeElementAt(0);
            probability *= weightedProbability(feature, category);
        }
        return probability;
    }

    private double categoryProbability(String item, String category){
        Integer categoryCount = (Integer)categoryCounter.get(category);
        double categoryProb = (double)categoryCount.intValue() / categoryCounter.size();
        return categoryProb * documentProbability(item, category);
    }

    public String classify(String item){
        final String DEFAULT = "";
        String category,
               best = "";
        double categoryProb,
               bestProbability,
               max = 0.0;
        Hashtable probabilities = new Hashtable();

        Enumeration categoryKeys = categoryCounter.keys();        
        while(categoryKeys.hasMoreElements()){
            category = (String)categoryKeys.nextElement();
            categoryProb = categoryProbability(item, category);
            probabilities.put(category, new Double(categoryProb));
            if(categoryProb > max){
                max = categoryProb;
                best = category;
            }
        }
        
        Enumeration probKeys = probabilities.keys();
        while(probKeys.hasMoreElements()){
            category = (String)probKeys.nextElement();
            if(category.equals(best))
                continue;
            categoryProb = ((Double)probabilities.get(category)).doubleValue();
            bestProbability = ((Double)probabilities.get(best)).doubleValue();
            if(categoryProb >= bestProbability)
                return DEFAULT;
        }
        return best;
    }    
}
