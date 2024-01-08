package codeForSharing;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Date;
import java.util.HashMap;
import java.util.TimeZone;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.RAMDirectory;

/** This class implements methods needed for building and searching the lucene index */

public class luceneIndex {

	Directory iIndex;
	IndexWriter index_writer;
	public static void main(String[] args) {
		

	}
	public luceneIndex() throws IOException
	{
		// Assuming the index will fit in memory
		iIndex = new RAMDirectory();

		 
	}
	public luceneIndex(String indexPath) throws IOException
	{
		// Assuming the index is read from desk
		iIndex = FSDirectory.open(Paths.get(indexPath));

		 
	}
	public luceneIndex(String indexPath,boolean threadW) throws IOException
	{
		// Assuming the index is read from desk
		if(threadW)
		{
			iIndex = FSDirectory.open(Paths.get(indexPath));
		
		WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer();
		IndexWriterConfig config = new IndexWriterConfig(analyzer);
		threadWriter= new IndexWriter(iIndex, config);
		}
		else
			iIndex = FSDirectory.open(Paths.get(indexPath));
		 
	}
	
	public  void addDocWithDetails_Thread(String docID, String title, String URL, String date, String type, String field,String body, String author) throws IOException {
	      
		  Document doc = new Document();
		  doc.add(new StringField("docID", docID, Field.Store.YES));
		  if(title==null) title="T";
		  doc.add(new StoredField("title", title));
		  if(URL==null) URL="u";
		  doc.add(new StoredField("url", URL));
		  doc.add(new StoredField("date",date));  
		  doc.add(new StoredField("type",type));  
		  doc.add(new StoredField("field",field));  
		  doc.add(new TextField("body", body, Field.Store.YES));
		  doc.add(new StoredField("author", author));
		  threadWriter.addDocument(doc);
		 // threadWriter.close();
		}
	
	public  ScoreDoc[] searchBody(String querytext) throws IOException, ParseException
	{
		boolean retry = true;
        while (retry)
        {
            try
            {
                retry = false;
              //  StandardAnalyzer analyzer = new StandardAnalyzer();
                WhitespaceAnalyzer analyzer = new WhitespaceAnalyzer(); //specially for dbpedia annotated index    		 
        		Query q = new QueryParser("body", analyzer).parse(querytext);
        		int hitsPerPage = 100;
        		IndexReader reader = DirectoryReader.open(iIndex);
        		IndexSearcher searcher = new IndexSearcher(reader);
        		TopDocs docs = searcher.search(q, hitsPerPage);
        		ScoreDoc[] hits = docs.scoreDocs;
        	//	printHits(hits);
        		return hits;        		
            }
            catch (Exception e)
            {
            	e.printStackTrace();
            	//System.out.println(e);
            	//trying this code
                // Double the number of boolean queries allowed.
                // The default is in org.apache.lucene.search.BooleanQuery and is 1024.
                String defaultQueries = Integer.toString(BooleanQuery.getMaxClauseCount());
                int oldQueries = Integer.parseInt(System.getProperty("org.apache.lucene.maxClauseCount", defaultQueries));
                int newQueries = oldQueries +1000;//* 2;
                
                System.setProperty("org.apache.lucene.maxClauseCount", Integer.toString(newQueries));
                //System.out.println("new queries changed to:"+newQueries);
                BooleanQuery.setMaxClauseCount(newQueries);
                retry = true;
                
                //return null;
            }
        }
		return null;
	}
	
}

