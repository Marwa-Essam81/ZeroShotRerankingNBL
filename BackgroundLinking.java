package codeForSharing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreDoc;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document.OutputSettings;
import org.jsoup.nodes.Entities.EscapeMode;
import org.jsoup.parser.Parser;
import org.jsoup.safety.Whitelist;
import org.tartarus.snowball.ext.EnglishStemmer;

import backgroundLinking.entrance;
import backgroundLinking.luceneIndex;



public class BackgroundLinking {

	public static void main(String[] args) {
		/** To retrieve an initial set of background links. You need to pass to this method:
* the index name, a file with the queries information 
* the location to a directory where each query has a text file with its title concatenated with its content after removing the html tags
* The output directory
* the output run file name
* the output log file name
* -1 (this parameter means that you do not limit the size of the search query to a specific number of terms.
* false (this parameter means that you will not idf as boost along with tf for term weighting during the retrieval process.

* Example: processQueriesTFIDF("indexName.index","QueriesInfo.txt", "QueriesTxt","OutputDir","Baseline_run.txt","Baseline_run_log.txt",-1,false);
*/

	}

	/**
	 * The following method creates a sql database that stores each article as a set of paragraph records after removing the html tags
	 */
	public static void createWBDatabase()
	{
		int fileno=14;
		try{
			Class.forName("com.mysql.jdbc.Driver");  
			
			
			
	    int count=0;
	    int indexed=0;
	    while(fileno<15)
	    {
	    	
	    	FileReader fileReader = new FileReader("V3CollectionFiles/TREC_WashingtonFile_"+fileno);
			

			BufferedReader bufferedReader = new BufferedReader(fileReader);
	        
		    String line=bufferedReader.readLine();
		    Connection con=DriverManager.getConnection(  
					"jdbc:mysql://localhost:3306/WpostDB","root","123456789");  
	    
		while(line!=null)
	    {
			
	    	String docid="";
	    	String pStmt="";
	    	try{
	    	count=count+1;
	    	if(count%1000==0) System.out.println(count);
	    	
	    	/***
	    	 * First Step Extract the kicker type
	    	 */
	    	String type="Normal";
	    	/**
	    	 * Extract the other data
	    	 */
	    	
	    	JSONObject	obj = (JSONObject) new JSONParser().parse(line);
	    	
	    	docid=(String)obj.get("id");
	    	
        	String title="T";
			try{ title=(String)obj.get("title");} catch(Exception te)  { }//te.printStackTrace();}
			if(title==null) { title="T";}
			String url="U";
			try{  url=(String)obj.get("article_url");} catch(Exception te)  { }//te.printStackTrace();}
			if(url==null) url="U";
			String author="A";
			try{  author=(String)obj.get("author");} catch(Exception te)  { }//te.printStackTrace();}
			if(author==null) author="A";
			long date=0;
			try{date=(Long)obj.get("published_date");} catch(Exception te) { }//te.printStackTrace();}
			
	        JSONArray contents=(JSONArray) obj.get("contents");
          
            /**
             * Round 1 ... get the title and date correctly
             */
            for(int i=0;i<contents.size();i++)
        	{
            	try {
            		JSONObject jobj=(JSONObject)contents.get(i);
            		String ctype="";
            		try{	ctype=(String)jobj.get("type");}catch(Exception te)  {}
    		
            				if(ctype.equals("title"))
            					if(title==null)
            					{
            						try{title=(String)jobj.get("content");
            						System.out.println(title);} catch(Exception te)  {}
            					}
            		else
            		if(date==0)
            			if(ctype.equals("date")) try{ date=(Long)obj.get("content");} catch(Exception te)  {}
            		else
                		if(ctype.equals("kicker"))
                		{
                			try{
                				type=(String)jobj.get("content");
                			}
                			catch(Exception te)  {}
                		}
    				}
            	
            	
            	/*
            	 * else if(ctype.equals("sanitized_html"))
            			contentStr=contentStr+" "+cStr;
            	 */
            	catch(Exception e)
            	{
            		System.out.print("error within document loop");
            		e.printStackTrace();
            		//not a type we are interested in
            	}
        	}
            /**
             * Saving the record of the document
             */
            pStmt="insert into Documents values ('"
            		+docid+
            		"','"
            		+ title.replace("'","\\'")
            		+ "','"+
            		type.replace("'","\\'")
            		+"','"+
            		url.replace("'","\\'")
            		+"','"+
            		author.replace("'","\\'")
            		+"',"+
            		Long.toString(date)+")";
            PreparedStatement preparedStmt = con.prepareStatement(pStmt);

		      preparedStmt.execute();
		      
		    /**
		     * Round 2: Saving the contents of paragraphs
		     */
		      int paragraphPos=1;
		      for(int i=0;i<contents.size();i++)
	        	{
	            	try {
	            		JSONObject jobj=(JSONObject)contents.get(i);
	            		String ctype=(String)jobj.get("type");
	            		if(ctype.equals("sanitized_html"))
	            		{
	            			String subtype=ctype=(String)jobj.get("subtype");
	            			
	            			if(subtype!=null&& subtype.equals("paragraph"))	            				
	            			{
	            				String content=(String)jobj.get("content");
	            				OutputSettings settings = new OutputSettings();
	            				settings.escapeMode(EscapeMode.base);
	            				String cleanHtml = Jsoup.clean(content, " ", Whitelist.none(), settings);
	            				cleanHtml=Parser.unescapeEntities(cleanHtml, false).replace("\\","\\\\").replace("'","\\'").replace("_","\\_").replace("%","\\%").replace("\"","\\'");
	            				if(!cleanHtml.replace(" ","").contentEquals(""))
	            				{
	            					String pid=""+docid+"_"+paragraphPos;
	            				
	            					preparedStmt = con.prepareStatement("insert into Contents values ('"
	            							+pid+
	            							"','"
	            							+docid+
	            		            		"','"
	            		            		+ cleanHtml
	            		            		+"',"+
	            		            		Integer.toString(paragraphPos)+")");
	            					paragraphPos=paragraphPos+1;
	            					preparedStmt.execute();
	            				} 
	            			}
	            		}
	            	}
	            	catch(NullPointerException e)
	            	{
	            	
	            	}
	            	catch(Exception e)
	            	{
	            		System.out.println("error in processing paragraphs of document:"+docid);
	            		System.out.println(preparedStmt);
	            		//e.printStackTrace
	            	}
	        	}  
	    	}
			catch(Exception e)
			{
				System.out.println("error adding document:"+docid);
				//System.out.println("-------:"+pStmt);
				e.printStackTrace();
			} 
	    	
	    	line=bufferedReader.readLine(); 
	    }
	    bufferedReader.close();
	    con.close();
	    fileno=fileno+1;
	    }
		}
		catch(Exception e)
		{
    		System.out.print("error ");
			e.printStackTrace();
		}
		
	}


	/**
	 * 	A method that applies preprocessing to the text (i.e stop words removal, lower casing,...)
	 * @param input
	 * @param lowercase
	 * @param stopWordsRemoval
	 * @param stopWords
	 * @param stem
	 * @param minTokenLength
	 * @return
	 */
	public static String preProcessStringStopWordsRemoved(String input,boolean lowercase,boolean stopWordsRemoval,ArrayList<String> stopWords, boolean stem,int minTokenLength)
	{
	// First Step is to filter the text to remove any remaining HTML content
	OutputSettings settings = new OutputSettings();
	settings.escapeMode(EscapeMode.base);
	String cleanHtml = Jsoup.clean(input, " ", Whitelist.none(), settings);
	cleanHtml=Parser.unescapeEntities(cleanHtml, false); // rempoving the &nbsp; resulted from parsing the html  

	if(lowercase) cleanHtml=cleanHtml.toLowerCase();
	String finaltxt="";
	if(stopWordsRemoval)
	{
		String[] substrings=cleanHtml.split(" ");

	for(int i=0;i<substrings.length;i++)
		if(!stopWords.contains(substrings[i].toLowerCase()))
				finaltxt=finaltxt+substrings[i]+" ";
	}
	else
		finaltxt=cleanHtml;
	// Now remove all non alphapetical characters and all extra spaces.
	finaltxt=finaltxt.trim().replaceAll("[^A-Za-z ]"," ").replaceAll("( )+"," ");
	//Make sure that no stop words are there after special character removal
	String finaltxt1="";
	if(stopWordsRemoval)
	{
		String[] substrings=finaltxt.split(" ");

	for(int i=0;i<substrings.length;i++)
		if(!stopWords.contains(substrings[i].toLowerCase()))
				finaltxt1=finaltxt1+substrings[i]+" ";
	}
	else
		finaltxt1=finaltxt;

	//Now removing all token less than min in length
	cleanHtml="";
	if(minTokenLength>0)
	{
		String[] substrings=finaltxt1.split(" ");
		for(int i=0;i<substrings.length;i++)
	    	if(substrings[i].length()>=minTokenLength)
	    		cleanHtml=cleanHtml+substrings[i]+" ";
		finaltxt1=cleanHtml;
	}

	// we need to apply stemming here if requested

	String output="";
	if(stem) {
		
		
		EnglishStemmer english = new EnglishStemmer();
	    String[] words = finaltxt1.split(" ");
	    for(int i = 0; i < words.length; i++){
	            english.setCurrent(words[i]);
	            english.stem();
	            output=output+english.getCurrent()+" ";
	    }
	}
	else
		output=finaltxt1;
	return output.strip();	

	}
	/**
	 * A method that applies background linking using the terms extracted from TF or TF-IDF
	 * @param lindex
	 * @param queryInfo
	 * @param queryDir
	 * @param Outdirectory
	 * @param outputfile
	 * @param loggerfile
	 * @param TFWords
	 * @param idf -- wether or not to include idf
	 */
	public static void processQueriesTFIDF(String lindex,String queryInfo,String queryDir,String Outdirectory,String outputfile,String loggerfile,int TFWords,boolean idf)
	{
		try
		{
			int errorInQueries=0;
			luceneIndex index=new luceneIndex(lindex);
			PrintWriter writer = new PrintWriter(Outdirectory+"/"+outputfile, "UTF-8");
			PrintWriter writerQuery = new PrintWriter(Outdirectory+"/query_"+outputfile, "UTF-8");
			
			PrintWriter writerLogger = new PrintWriter(Outdirectory+"/"+loggerfile, "UTF-8");
			
			/** loading stop words list 
			 */
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
		        
			/**
	    	 * loading ids of documents to correct missing dates in collection file
	    	 */
			HashMap <String,Long> idDates =new HashMap <String,Long>();
			try
	    	{
			 fileReader = new FileReader("DocIdsDates.txt");

			 bufferedReader = new BufferedReader(fileReader);
	        
		    String line=bufferedReader.readLine();
		   
		  
		    while(line!=null)
		    {
		    	String[] str=line.split(",");
		    	idDates.put(str[0],Long.parseLong(str[1]));
		    	line=bufferedReader.readLine();
		    }

	    	}
	    	catch(Exception te) {te.printStackTrace();}
			
			
			 fileReader = new FileReader(queryInfo);
			 bufferedReader = new BufferedReader(fileReader);
	        
		    String line=bufferedReader.readLine();
		   
		    Long TotalTime=0l;
		    int qcount=0;
		    long totalextractiontime=0l;
		    while(line!=null)
		    {
		    	String[] str=line.split("#");
		    	String topicNo=str[0];
		    	Long qDate=Long.parseLong(str[1]);
		    	String qTitle=str[2];
		    	String searchQuery=getQueryBody(queryDir+"/"+topicNo+".txt");
		    
		    	searchQuery=preProcessStringStopWordsRemoved(searchQuery, true, true, stopwords, false,2);
		    	long starttimeE=System.currentTimeMillis();
		    	searchQuery=getTermsTFIDF(searchQuery,idf,TFWords,lindex);
		    	totalextractiontime=totalextractiontime+(System.currentTimeMillis()-starttimeE);
		    	
		    		qTitle=preProcessStringStopWordsRemoved(qTitle, true, false, stopwords, false,0);
		    	
		    	
		    	//
		    	ArrayList<String> DocSignatures=new ArrayList<String>();
		    	DocSignatures.add(qTitle);

		    	writerQuery.println(searchQuery);
		    	qcount=qcount+1;
		    	long starttime=System.currentTimeMillis();

		    	ScoreDoc[] hits=index.searchBody(searchQuery.trim());
		    	
		    	Long endtime=System.currentTimeMillis();
		    	Long Result=(endtime-starttime);
		    	TotalTime=TotalTime+Result;
		    	writerLogger.println(topicNo+" "+Result);
		    	IndexReader reader = DirectoryReader.open(index.iIndex);
		    	IndexSearcher searcher = new IndexSearcher(reader);
		    	

		    	int count=0;
		    	// Now checking every hit to see if it can be added to our result set:
		    	for(int i=0;i<hits.length;i++) {
		    	    int docId = hits[i].doc;
		    	    Document d = searcher.doc(docId);
		    	    String docid=d.get("docID");
		    	    Long docDate=(long)0;
		    	    try{docDate=idDates.get(docid);} catch(Exception e) {e.printStackTrace();}
		    	   // System.out.println(docDate);
		    	    String field=d.get("field");
		    	    String type=d.get("type");
		    	    long datedifference=0;
		    	    if(docDate!=null)
		    	  	   	datedifference=qDate-docDate;
		    		 
		    	    if(field.equals("Opinion")||field.equals("Editor")||field.equals("PostView")) continue;
		    	    if(type.equals("Opinion")||type.equals("Editor")||type.equals("PostView")) continue;
		    		String signature=preProcessStringStopWordsRemoved(d.get("title"), true, false, stopwords, false,0);
		    				
		    	    if(!DocSignatures.contains(signature))  //only if this document was not added before
		    	    if(datedifference>0&&(docDate!=null)&&(docDate!=0)) // only if the retrieved article is published before the current article, it can be added to the result
		    	       	{
		    	    	//System.out.println("Document before: "+docid);
		    	    	count=count+1;
		    	    	
		    	    	writer.println(topicNo+" Q0 "+docid+" 0 "+hits[i].score+" QU_KTR");
		    	    	if(count==100)
		    	    		break;
		    	       	}
		    	    
		    	}
		    	if(count<100)
		    	{
		    		//System.out.println("error in Topic"+topicNo);
		    		errorInQueries=errorInQueries+1;
		    	}
		    		
		    	line=bufferedReader.readLine();
		    }
		   // System.out.print(TotalTime+" ");	
		   System.out.println("TF Extraction Time "+totalextractiontime+" "+idf);	

		    
		  //  System.out.println(TotalTime);	
		  // System.out.println("error in query "+errorInQueries);	
		    writer.close();
		    writerLogger.close();
		    writerQuery.close();
		}
		catch(Exception te) {te.printStackTrace();}
	}	    
}
