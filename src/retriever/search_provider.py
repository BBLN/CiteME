from utils.semantic_scholar import SemanticScholarAPI, SemanticScholarWebSearch
from utils import PaperSearchResult
from typing import List
from retriever.pinecone_db import get_pinecone_client, get_or_create_index
from retriever.specter_embedding import Specter2AdhocQuery

class SearchProvider:
    def __call__(self, search_term: str, year: str) -> List[PaperSearchResult]:
        raise NotImplementedError("Subclasses must implement this method")


class SemanticScholarSearchProvider(SearchProvider):
    def __init__(
        self,
        fieldsOfStudy: str = "Computer Science",
        limit: int = 10,
        # sort: str = "citationCount:desc",
        only_open_access: bool = False,
    ):
        # These fields are required for the PaperSearchResult model
        self.fields = (
            "paperId,title,authors,venue,year,citationCount,abstract,openAccessPdf"
        )
        self.fieldsOfStudy = fieldsOfStudy
        self.limit = limit
        self.only_open_access = only_open_access
        # self.sort = sort
        self.s2api = SemanticScholarAPI()

    def citation_count_search(
        self, query: str, year: str | None, max_search_limit: int = 100, bulk: bool = False
    ) -> List[PaperSearchResult]:
        papers: List[PaperSearchResult] = []
        if bulk:
            tmp_papers = self.s2api.bulk_search(query, self.fields, self.fieldsOfStudy)
            papers += [PaperSearchResult(**paper) for paper in tmp_papers["data"]]
        else:
            for offset in range(0, max_search_limit, 100):
                tmp_papers = self.s2api.relevance_search(
                    query,
                    self.fields,
                    self.fieldsOfStudy,
                    year=year,
                    limit=min(100, max_search_limit - offset),
                    offset=offset,
                    only_open_access=self.only_open_access,
                )
                if "data" not in tmp_papers:
                    break
                papers += [PaperSearchResult(**paper) for paper in tmp_papers["data"]]

        papers = sorted(papers, key=lambda x: x.citationCount, reverse=True)
        return papers[: self.limit]

    def __call__(self, query: str, year: str | None = None) -> List[PaperSearchResult]:
        # papers = self.s2api.bulk_search(query, self.fields, self.fieldsOfStudy, self.sort)
        papers = self.s2api.relevance_search(
            query,
            self.fields,
            self.fieldsOfStudy,
            year,
            self.limit,
            only_open_access=self.only_open_access,
        )
        if "data" not in papers:
            return []
        papers = [PaperSearchResult(**paper) for paper in papers["data"]]
        return papers


class SemanticScholarWebSearchProvider(SearchProvider):
    def __init__(
        self,
        fieldsOfStudy: str = "Computer Science",
        limit: int = 10,
        # sort: str = "citationCount:desc",
        only_open_access: bool = False,
    ):
        # These fields are required for the PaperSearchResult model
        self.fields = (
            "paperId,title,authors,venue,year,citationCount,abstract,openAccessPdf"
        )
        self.fieldsOfStudy = fieldsOfStudy
        self.limit = limit
        self.only_open_access = only_open_access
        # self.sort = sort
        self.s2api = SemanticScholarWebSearch()

    def citation_count_search(
        self, query: str, year: str | None
    ) -> List[PaperSearchResult]:
        papers = self.s2api.web_search(
            query,
            fields=self.fields,
            fields_of_study=(
                self.fieldsOfStudy.replace(" ", "-").lower()
                if self.fieldsOfStudy
                else None
            ),
            sort="total-citations",
            cutoff_year=year.replace("-", "") if year else None,
            limit=self.limit,
            only_open_access=self.only_open_access,
        )
        papers = [PaperSearchResult(**paper) for paper in papers]
        return papers[: self.limit]

    def __call__(self, query: str, year: str | None = None) -> List[PaperSearchResult]:
        # papers = self.s2api.bulk_search(query, self.fields, self.fieldsOfStudy, self.sort)
        papers = self.s2api.relevance_search(
            query,
            self.fields,
            self.fieldsOfStudy,
            year,
            self.limit,
            only_open_access=self.only_open_access,
        )
        papers = [PaperSearchResult(**paper) for paper in papers]
        return papers

class RAGProvider(SearchProvider):
    def __init__(
        self,
        index_name="semanticscholar-index-specter2-11-16-2024"
    ):
        self.embedding_model = Specter2AdhocQuery()
        self.pc = get_pinecone_client()
        self.index = get_or_create_index(self.pc, index_name, self.embedding_model.embedding_dim())

    def _embed_query(self, query: str):
        return self.embedding_model.embed_parallel([query]).squeeze()
    
    def _query_index(self, query: str, top_k: int = 10):
        query_embedding = self._embed_query(query)
        return self.index.query(query_embedding.tolist(), namespace="semanticscholar-metadata", top_k=top_k, include_metadata=True)
    
    def _map_metadata(self, metadata):
        return PaperSearchResult(
            paperId=metadata["paperId"],
            title=metadata["title"],
            citationCount=metadata["citationCount"],
            year=metadata["year"],
            authors=[],
            venue=None,
            abstract=metadata['abstract'] if 'abstract' in metadata else None,
            openAccessPdf=None
        )

    def citation_count_search(
        self, query: str, year: str | None
    ) -> List[PaperSearchResult]:
        papers = self._query_index(query, 50)
        papers = [self._map_metadata(paper['metadata']) for paper in papers['matches']]
        papers = sorted(papers, key=lambda x: x.citationCount, reverse=True)
        return papers[:10]
    
    def __call__(self, query: str, year: str | None = None) -> List[PaperSearchResult]:
        papers = self._query_index(query)
        papers = [self._map_metadata(paper['metadata']) for paper in papers['matches']]
        return papers
