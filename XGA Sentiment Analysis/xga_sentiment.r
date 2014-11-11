xga_sentiment<-function(unstructured_text) {
	data <- apply(as.matrix(unstructured_text),1,paste,collapse=" ")
	data_corpus <- Corpus(VectorSource(data))
	data_matrix <- DocumentTermMatrix(data_corpus,control=list(language="english", removeNumbers=TRUE, removePunctuation=TRUE, removeSparseTerms=0, removeStopwords=TRUE, stemWords=FALSE, stripWhitespace=TRUE, toLower=TRUE));
   	lex_dict <- read.csv("data/lex_dict.csv",header=FALSE)
	lex_count <- list(positive=length(which(lex_dict[,2]=="positive")),negative=length(which(lex_dict[,2]=="negative")),average=length(which(lex_dict[,2]=="average")),
	total=nrow(lex_dict))
	
	docs <- c()

	for (i in 1:nrow(data_matrix)) {
		scores <- list(positive=0,negative=0,average=0)
		words <- findFreqTerms(data_matrix[i,],lowfreq=1)
		for (j in words) {
			match <- pmatch(j,lex_dict[,1],nomatch=0)
			if (match > 0) {
				sentiment <- as.character(lex_dict[match,][[2]])
				individual_score <- (lex_dict[match,][[3]])
				tot_count <- counts[[sentiment]]
				score <- abs(log(individual_score*1.0/tot_count))
				scores[[sentiment]] <- scores[[sentiment]]+score
			}		
		}
			for (k in names(scores)) 
				{
				naive_score <- abs(log(counts[[k]]/counts[["total"]]))
				scores[[key]] <- scores[[key]]+naive_score
				} 
        final_sent <- names(scores)[which.max(unlist(scores))]
		docs <- rbind(docs,c(scores$positive,scores$negative,final_sent))
		}
	colnames(docs) <- c("POSITIVE","NEGATIVE","FINAL_SENTIMENT")
	return(docs)
}

