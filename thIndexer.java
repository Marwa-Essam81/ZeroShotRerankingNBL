package codeForSharing;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document.OutputSettings;
import org.jsoup.nodes.Entities.EscapeMode;
import org.jsoup.parser.Parser;
import org.jsoup.safety.Whitelist;
import org.tartarus.snowball.ext.EnglishStemmer;

public class thIndexer 
extends Thread {
	public String fileName="";
	luceneIndex index;
	ArrayList<String> stopwords;
	boolean stopwordsremoval;
	boolean stem;
	int minTokenLength;
	
	public thIndexer(String fileName,luceneIndex index,ArrayList<String> stopwords,boolean stopwordsremoval,boolean stem,int minTokenLength)
	{
		this.fileName=fileName;
		this.index=index;
		this.stopwords=stopwords;
		this.stopwordsremoval=stopwordsremoval;
		this.stem=stem;
		this.minTokenLength=minTokenLength;
	}
    public void run()
    {
        try {
            // Displaying the thread that is running
            System.out.println(
                "Thread " + Thread.currentThread().getId()
                + " is running");
            //createV4Index(index, fileName, stopwords, stopwordsremoval, stem, minTokenLength);
            createIndex(index, fileName, stopwords, stopwordsremoval, stem, minTokenLength);
        }
        catch (Exception e) {
            // Throwing an exception
            System.out.println("Exception is caught");
        }
    }
    public static void createIndex(luceneIndex index, String collectionfile,ArrayList<String> stopwords,boolean stopwordsremoval,boolean stem,int minTokenLength)
    {
    	try
    	{
    	       
    		FileReader fileReader = new FileReader(collectionfile);

    		BufferedReader bufferedReader = new BufferedReader(fileReader);
            
    	    String line=bufferedReader.readLine();
    	    int count=0;
    	    int indexed=0;
    	    while(line!=null)
    	    {
    	    	count=count+1;
    	      if(count%1000==0) {System.out.println(collectionfile+" : "+count);System.out.println(count);}

    	    	String type="Normal";
    	    	String field="";
    	    	if(line.contains("\"kicker\": "	+ "\"Opinion\""))
    	    		field="Opinion";
    	    	else 
    	    	if(line.contains("\"kicker\": "	+ "\"Opinions\""))
    	    		field="Opinions";
    	    	else
    	    	if(line.contains("\"kicker\": "	+ "\"Letters to the Editor\""))
    	    		field="Editor";
    	    	else 
    	    	if(line.contains("\"kicker\": "	+ "\"The Post's View\""))
    	    		field="PostView";	
    	   
    	    	if(line.contains("content\": \"Opinion\""))
    	    		type="Opinion";
    	    	else
    	    	if(line.contains("content\": \"Opinions\""))
    		    	type="Opinions";
    	    	else
    	    	if(line.contains("content\": \"Letters to the Editor\""))
    	    		type="Editor";
    	    	else
    	    	if(line.contains("content\": \"The Post's View\""))
    	    		type="PostView";
    	    	
    	    	/*
    	    	 * Starting extracting the contents of the object
    	    	 */
    	    	try
    	    	{
    	    	
    	    	JSONObject	obj = (JSONObject) new JSONParser().parse(line);
    	    	
            	String docid=(String)obj.get("id");
            	String title="T";
    			try{ title=(String)obj.get("title");} catch(Exception te)  {}
    			if(title==null) title="T";
    			String url="U";
    			try{  url=(String)obj.get("article_url");} catch(Exception te)  {}
    			if(url==null) url="U";
    			String author="A";
    			try{  author=(String)obj.get("author");} catch(Exception te)  {}
    			if(author==null) author="A";
    			long date=0;
    			if(line.contains("publish_date"))
    					try{date=(Long)obj.get("publish_date");} 
    						catch(Exception td) {
    						}
    			else
    				if(line.contains("published_date"))
    					try{date=(Long)obj.get("published_date");} 
    					catch(Exception te) 
    					{}
    			
    	//		System.out.println(docid);
                JSONArray contents=(JSONArray) obj.get("contents");
                String contentStr="";
                boolean opinion=false;
                for(int i=0;i<contents.size();i++)
            	{
                	try {
                		JSONObject jobj=(JSONObject)contents.get(i);
                		String ctype="";
                		try{	ctype=(String)jobj.get("type");}catch(Exception te)  {}
                		
                	
                		String cStr=(String)jobj.get("content");
        		
                		if(ctype.equals("title")||ctype.equals("sanitized_html"))
                			contentStr=contentStr+" "+cStr;
                		if(ctype.equals("title")&&title.contentEquals("T"))
                			title=cStr;
                		if(ctype.equals("date")&&(date==0))
                		{
                			String sdate=(String)jobj.get("content");
    						if(sdate.contains("Z"))
    							sdate=sdate.substring(0,sdate.indexOf("Z"));
    						if(!sdate.contains("."))
    							sdate=sdate+".000";
    						try{
    							LocalDateTime ldate = LocalDateTime.parse(sdate, DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS"));
    						
    						ZonedDateTime zdt = ldate.atZone(ZoneOffset.UTC);
    						
    						date=zdt.toInstant().toEpochMilli();
    						}
    						catch(Exception ed)  {System.out.println("error storing date for document id:"+docid);}
    						//te.printStackTrace();
    					
                		}
                	}
                	catch(Exception e)
                	{
                		//not a type we are interested in
                	}
            	}
                /**
                 * saving the contents found in lucene
                 */
                String body=preProcessString(contentStr,stopwordsremoval,stopwords,stem,minTokenLength);
               
                index.addDocWithDetails_Thread(docid, title, url, Long.toString(date), type, field, body, author);
                indexed=indexed+1;
                //  printline(docid, title, url, Long.toString(date), type, field, body, author);
    	    	}
            	catch(Exception e)
            	{
            		System.out.println("Unknown error at line: "+line);
            		e.printStackTrace();
            	}
    		//	System.out.println("continue?");

    	   	//	int x= System.in.read();
    		//	if(x==0) break;
    	   		line=bufferedReader.readLine();
    	    }
    	    index.commitWriter();
    	    System.out.println("Total Processed: "+count);
    	    System.out.println("Total Indexed: "+indexed);
    	    System.out.println("Total Processed: "+count);
    	    System.out.println("Total Indexed: "+indexed);
    	    //logger.close();
    	    bufferedReader.close();
    	}
    	catch(Exception e)
    	{
    		e.printStackTrace();
    	}

    }
    
    
    public static String preProcessString(String input,boolean stopWordsRemoval,ArrayList<String> stopWords, boolean stem,int minTokenLength)
    {
    	// First Step is to filter the text to remove any remaining HTML content
    	OutputSettings settings = new OutputSettings();
        settings.escapeMode(EscapeMode.base);
        String cleanHtml = Jsoup.clean(input, " ", Whitelist.none(), settings);
        cleanHtml=Parser.unescapeEntities(cleanHtml, false).toLowerCase(); // rempoving the &nbsp; resulted from parsing the html  
        
        String finaltxt="";
    	if(stopWordsRemoval)
    	{
    		String[] substrings=cleanHtml.split(" ");
    	
        for(int i=0;i<substrings.length;i++)
        	if(!stopWords.contains(substrings[i]))
        			finaltxt=finaltxt+substrings[i]+" ";
    	}
    	else
    		finaltxt=cleanHtml;
    	// Now remove all non alphapetical characters and all extra spaces.
    	finaltxt=finaltxt.trim().replaceAll("[^A-Za-z ]"," ").replaceAll("( )+"," ");
    	
    	//Now removing all token less than min in length
    	cleanHtml="";
    	if(minTokenLength>0)
    	{
    		String[] substrings=finaltxt.split(" ");
    		for(int i=0;i<substrings.length;i++)
    	    	if(substrings[i].length()>=minTokenLength)
    	    		cleanHtml=cleanHtml+substrings[i]+" ";
    		finaltxt=cleanHtml;
    	}
    	
    	// we need to apply stemming here if requested
    	
    	String output="";
    	if(stem) {
    		
    		
    		EnglishStemmer english = new EnglishStemmer();
    	    String[] words = finaltxt.split(" ");
    	    for(int i = 0; i < words.length; i++){
    	            english.setCurrent(words[i]);
    	            english.stem();
    	            output=output+english.getCurrent()+" ";
    	    }
    	}
    	else
    		output=finaltxt;
    	return output;	
    	
    }

}
