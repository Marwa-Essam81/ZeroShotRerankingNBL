package codeForSharing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;

public class indexer {
	/**
	 * In the main method, we initiate threads that executes the indexing procedure after splitting the big collection file into multiple files for fast indexing
	 */
	
	public static void main(String[] args) {
		try {
			System.out.println("Starting the indexing process");
			
			luceneIndex indexR=new luceneIndex("V3Index_Stemmed.index",false);
			IndexReader reader=DirectoryReader.open(indexR.iIndex);
				
			
			
			splitCollectionFile("TREC_Washington_Post_collection.v3.jl","DataFiles"); //Assuming here that DataFiles is the output direcory is : DataFiles
			//load stop words list 
			ArrayList<String> stopwords=new ArrayList<String>();
	   		 FileReader fileReader = new FileReader("StopWordsSEO.txt");

	   	     BufferedReader bufferedReader = new BufferedReader(fileReader);
	   	        
	   	        String lineInput=bufferedReader.readLine();
	   	        while(lineInput!=null)
	   	        {
	   	        	stopwords.add(lineInput.trim());
	   	        	lineInput=bufferedReader.readLine();
	   	        }
	   	        bufferedReader.close();
	   	        
			int n = 15; // Number of threads
			luceneIndex index=new luceneIndex("V3Index_Stemmed.index",true);
	        for (int i = 14; i < n; i++) {
	            thIndexer object
	                = new thIndexer("DataFiles/Part_"+i+".txt",index,stopwords,true,true,2); 		
	            object.start();
	        }
	        
			index.commitWriter();
			
			
			//Document d=index.searchById("f831cae6-bfa4-11e1-9ce8-ff26651238d0");
			//System.out.println(d.get("rawBody"));
			*/
			 }	// TODO Auto-generated method stub

	}
	/**
	 * A method that splits the big collection file into parts to prepare it for threading.
	 * @param fileurl
	 * @param outputDir
	 */
	public static void splitCollectionFile(String fileurl,String outputDir)
	{
		int fileNo=1;
		try
		{
			PrintWriter  writer= new PrintWriter(outputDir+"/Part_"+fileNo+".txt", "UTF-8");
			
			 FileReader fileReader = new FileReader(fileurl);

		     BufferedReader bufferedReader = new BufferedReader(fileReader);
		        
		    String lineInput=bufferedReader.readLine();
		    int count=0;
		        while(lineInput!=null)
		        {
		        	writer.println(lineInput);
		        	count=count+1;
		        	if(count%50000==0)
		        	{
		        		System.out.println("Done File:"+fileNo);
		        		writer.close();
		        		fileNo=fileNo+1;
		        		 writer= new PrintWriter("DataFiles/V4_"+fileNo+".txt", "UTF-8");
		        	}
		        	lineInput=bufferedReader.readLine();
		        }
		        writer.close();
		        System.out.println("Processed Articles:"+count);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		
	}


}
