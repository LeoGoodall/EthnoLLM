# EthnoLLM
_LLMs struggle with ethnographic text annotation_

All python scripts used to handle data and model inferences are in /src. These may need to be moved to home directory to call the /data appropriately. Batch API calling to OpenAI, Anthropic, and Perplexity are omitted (though can be provided upon reasonable request). 

Ethnographic texts are excluded given a license from eHRAF is required for use.

Rscripts involve all statistical analyses used and results are stored in appropriately-named folders.

All model inferences are stored in /all and /synchrony folders. _Note_. The manuscript distinguishes between the "Morphospace" and "synchrony" datasets. In the code, these are referred to as the "all" and "synchrony" folders. To be clear, "all" does not refer to Morphospace+synchrony, but just Morphospace dataset.

To replicate these results:

Run analysis_all.py and analysis_synchrony.py for all models (GPU and HPC handling are omitted).
Submit API requests (batch or otherwise)
Run all Rscripts.
