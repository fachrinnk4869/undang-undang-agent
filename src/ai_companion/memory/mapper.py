from ai_companion.memory.memory_template import ContextMemory, Memory


class ChromaMemoryMapper:
    def map_results(self, results):
        """Return a mapping of the context memory for storage."""
        memories = []
        for i in range(len(results["ids"][0])):
            hit = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                **results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }

            mem = Memory(
                text=hit.get('document', ''),
                metadata={
                    'id': hit.get('id'),
                },
                score=hit.get('distance')
            )
            memories.append(mem)
        return memories


class ChromaContextMapper:
    def map_results(self, results):
        """Return a mapping of the context memory for storage."""
        memories = []
        for i in range(len(results["ids"][0])):
            hit = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                **results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }

            mem = ContextMemory(
                text=hit.get('document', ''),
                metadata={
                    'id': hit.get('id'),
                    'file_name': hit.get('file_name'),
                    'pasal': hit.get('pasal'),
                    'tahun': hit.get('tahun'),
                },
                score=hit.get('distance')
            )
            memories.append(mem)
        return memories
