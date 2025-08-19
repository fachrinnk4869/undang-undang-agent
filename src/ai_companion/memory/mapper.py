from typing import List
from ai_companion.memory.memory_template import ContextMemory, Memory


class ChromaMemoryMapper:
    def map_results(self, results) -> List[Memory]:
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
    def map_results(self, results) -> List[ContextMemory]:
        """Return a mapping of the context memory for storage."""
        memories = []
        try:
            for i in range(len(results["ids"][0])):
                # print("results", results)
                hit = {
                    "id": results["ids"][0][i],
                    **results["metadatas"][0][i],
                    "distance": results["distances"][i]
                }
                # print(f"Hit: {hit}")

                mem = ContextMemory(
                    text=hit.get('text', ''),
                    metadata={
                        'id': hit.get('id'),
                        'file_name': hit.get('file_name'),
                        'nomor': hit.get('nomor'),
                        'tahun': hit.get('tahun'),
                    },
                    score=hit.get('distance')
                )
                memories.append(mem)
        except KeyError as e:  # handle error [0] untuk yang just filter
            for i in range(len(results["ids"])):
                # print("results", results)
                # print("results", results['metadatas'][i])
                hit = {
                    "id": results["ids"][i],
                    **results["metadatas"][i]
                }
                # print(f"Hit: {hit}")

                mem = ContextMemory(
                    text=hit.get('text', ''),
                    metadata={
                        'id': hit.get('id'),
                        'file_name': hit.get('file_name'),
                        'nomor': hit.get('nomor'),
                        'tahun': hit.get('tahun'),
                    },
                    score=hit.get('distance')
                )
                memories.append(mem)
        return memories
