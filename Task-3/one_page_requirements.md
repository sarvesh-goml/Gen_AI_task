# One-Pager Mapping Document

## AI Shopping Assistant

---

## 1. Decision Gate

| Capability       | Decision    | Reason                                                                                                                                                        | Actual v1 Choice                                           |
| ---------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **RAG**          | **YES**     | Product information, specifications, availability, policies, and catalog data can change frequently and should not be memorized by the LLM.                   | Retrieval-Augmented Generation using PostgreSQL + pgvector |
| **Agentic AI**   | **YES**     | The assistant should not only recommend products but also perform controlled shopping operations such as searching, comparing, managing carts, and wishlists. | One controlled tool-using shopping agent                   |
| **Fine-Tuning**  | **NO — v1** | General-purpose LLM reasoning combined with product RAG and structured tools is sufficient for the initial version.                                           | No fine-tuning                                             |
| **Distillation** | **NO — v1** | Initial usage does not justify the additional training and model-management complexity.                                                                       | No distillation                                            |

### Overall Decision

Build a **RAG-powered single shopping agent with controlled shopping tools**.

The assistant should be able to understand natural-language shopping requests, retrieve relevant products, compare alternatives, explain recommendations, and perform approved shopping actions.

Avoid multi-agent orchestration and unnecessary infrastructure in v1.

---

# 2. RAG

## Decision

Use RAG to ground the LLM in **current product and shopping knowledge** instead of expecting the model to memorize product specifications, prices, policies, or inventory information.

## Knowledge Sources

The knowledge base can contain:

* Product descriptions
* Product specifications
* Product categories
* Product features
* Product reviews
* Product FAQs
* Shipping information
* Return and refund policies
* Warranty information
* Store policies
* Buying guides
* Product comparison information

Example product categories:

```text
knowledge/
├── electronics/
├── laptops/
├── smartphones/
├── headphones/
├── cameras/
├── home-appliances/
├── fashion/
├── footwear/
├── accessories/
└── buying-guides/
```

Each knowledge document should contain:

* Product/category title
* Product description
* Specifications
* Features
* Advantages
* Limitations
* Use cases
* Price information where applicable
* Availability information where applicable
* Warranty information
* Source/category metadata

### Example

```text
Product: Dell XPS 15

Category: Laptop

Description:
Premium laptop designed for professional and creative workloads.

Specifications:
- Processor: Intel Core Ultra 7
- RAM: 16 GB
- Storage: 1 TB SSD
- Display: 15.6-inch
- GPU: NVIDIA RTX

Best For:
- Software development
- Content creation
- Professional workloads

Limitations:
- Higher price
- Heavier than ultraportable laptops

Metadata:
category=laptop
brand=dell
price_range=premium
use_case=productivity
```

---

## RAG Design

| Component       | Decision                                                                          |
| --------------- | --------------------------------------------------------------------------------- |
| Document format | Markdown / structured product data                                                |
| Chunking        | Section-based chunking; keep product specifications and related features together |
| Embedding model | `text-embedding-3-small`                                                          |
| Vector database | PostgreSQL with pgvector                                                          |
| Retrieval       | Top-k semantic similarity + metadata filtering                                    |
| Filtering       | Category, brand, price range, rating, features, use case                          |
| Generation      | LLM receives retrieved product information as grounded context                    |
| Citations       | Product name and source/catalog information included where appropriate            |

### Retrieval Flow

```text
User Shopping Request
        ↓
Understand Intent + Constraints
        ↓
Create Query Embedding
        ↓
PostgreSQL + pgvector
        ↓
Retrieve Relevant Products
        ↓
Apply Structured Filters
        ↓
LLM + Retrieved Product Context
        ↓
Recommendation / Comparison / Explanation
```

### Important Design Principle

RAG should handle **semantic product discovery**, while structured database queries should handle exact constraints.

For example:

> "Find me a laptop under ₹80,000 with 16 GB RAM and an RTX GPU."

The system should not rely only on vector similarity.

It should combine:

```text
Semantic Search
       +
Structured Filtering
       ↓
Relevant Products
```

This provides more reliable shopping recommendations.

### Reason for PostgreSQL + pgvector

PostgreSQL can store:

* Product information
* Product metadata
* User preferences
* Cart data
* Wishlist data
* Product embeddings
* Shopping history

Using pgvector avoids introducing a separate vector database such as Pinecone, Weaviate, or Milvus.

---

# 3. Agentic AI

## Decision

Use **one controlled shopping agent with explicit tools**.

The agent should be able to understand the user's shopping intent, search products, retrieve product information, compare alternatives, and perform approved shopping operations.

## Tools

| Tool                    | Risk     | Execution                                |
| ----------------------- | -------- | ---------------------------------------- |
| `search_products`       | Low      | Automatic                                |
| `get_product`           | Low      | Automatic                                |
| `search_knowledge_base` | Low      | Automatic                                |
| `compare_products`      | Low      | Automatic                                |
| `get_product_reviews`   | Low      | Automatic                                |
| `get_cart`              | Low      | Automatic                                |
| `add_to_cart`           | Medium   | Confirmation depending on implementation |
| `remove_from_cart`      | Medium   | Confirmation depending on implementation |
| `add_to_wishlist`       | Low      | Automatic                                |
| `update_cart_quantity`  | Medium   | Confirmation                             |
| `apply_coupon`          | Medium   | Confirmation                             |
| `checkout`              | **High** | Require explicit confirmation            |
| `place_order`           | **High** | Require explicit confirmation            |

The agent must **never directly access the database**.

It can only interact with the shopping system through defined service/API tools.

---

## Planning Approach

### Decision: Use a Simple Tool-Calling Loop

```text
User Request
     ↓
Shopping Agent
     ↓
Understand Intent
     ↓
Decide:
Answer / Search / Compare / Use Tool
     ↓
Tool Execution
     ↓
Tool Result
     ↓
Shopping Agent
     ↓
Final Response
```

### Example

User:

> "I need wireless headphones under ₹5,000 for travel."

Agent:

```text
1. Understand budget = ₹5,000
2. Understand category = headphones
3. Understand use case = travel
4. Search products
5. Filter products
6. Compare relevant products
7. Recommend top options
```

The agent should explain **why** a product is recommended instead of simply returning search results.

---

# Shopping Memory

## Decision

Use **short-term shopping context** and selected user preferences.

The assistant can remember information relevant to the current shopping task such as:

* Current conversation
* Current product comparison
* Current cart
* Current wishlist
* User's stated budget
* Preferred brands
* Required features
* Product preferences expressed during the session

Example:

```text
User:
"I want a laptop under ₹80,000."

Later:
"What about this one?"

Agent:
Understands that "this one" refers to the product currently being discussed.
```

A separate long-term vector memory is **not required in v1**.

Long-term personalization can be introduced later using explicit user preference storage.

---

# Model Routing

## Decision

Use one capable LLM for v1.

### Primary LLM

**OpenAI GPT-5 mini**

### Embeddings

**text-embedding-3-small**

No separate planner model or multiple-LLM routing layer is required initially.

The LLM handles:

* Intent understanding
* Query interpretation
* Product recommendation reasoning
* Comparison explanation
* Tool selection
* Natural-language responses

Structured application logic handles:

* Price filtering
* Availability
* Product IDs
* Cart operations
* Order operations
* Database queries

---

# MCP

## Decision

**NO — v1**

The shopping tools are internal application functions/APIs.

MCP would introduce another integration layer without providing enough benefit for the initial capstone.

MCP can be considered later if the assistant needs to connect to multiple external commerce systems or third-party shopping services.

---

# Multi-Agent Architecture

## Decision

**NO — v1**

Use one shopping agent with multiple controlled tools.

Do not create separate agents for:

* Product search
* Recommendation
* Comparison
* Cart management
* Checkout
* Customer support

### Reason

A multi-agent architecture would increase:

* Orchestration complexity
* Debugging difficulty
* Latency
* Tool coordination complexity
* Evaluation complexity

without providing a necessary benefit for the initial system.

---

# 4. Fine-Tuning

## Decision

**Not implemented in v1.**

## Reason

The first version should establish a strong baseline using:

* Prompt engineering
* Product RAG
* Structured product filtering
* Tool calling
* Guardrails
* Evaluation

A general-purpose LLM should be capable of handling the initial shopping conversations when supplied with accurate product information.

The project initially does not have enough high-quality examples of:

* User shopping requests
* Ideal product recommendations
* Product comparisons
* Preference-based recommendations
* Successful shopping conversations

to justify fine-tuning.

## Future

After collecting sufficient validated shopping interaction data, LoRA/QLoRA fine-tuning can be evaluated for:

* Personalized recommendation style
* Product categorization
* Intent classification
* Query rewriting
* Consistent recommendation behavior
* Domain-specific shopping conversations

Fine-tuning should only be introduced after measuring that **prompting + RAG + tools** is insufficient.

---

# 5. Distillation

## Decision

**Not implemented in v1.**

## Reason

The initial capstone does not have enough inference volume or latency pressure to justify model distillation.

Distillation would introduce:

* Training complexity
* Evaluation complexity
* Student-model management
* Additional deployment complexity

without providing enough immediate benefit.

## Future

Distillation can be considered if production usage creates significant:

* LLM cost
* Latency
* Throughput requirements

A smaller student model could then handle simpler tasks such as:

* Product categorization
* Intent classification
* Query rewriting
* Basic product filtering
* Simple FAQ responses

while the larger model handles complex recommendation and reasoning tasks.

---

# 6. LLMOps

## Decision

Implement **lightweight but real LLMOps** rather than introducing a large MLOps/LLMOps platform.

| Area                      | v1 Design                                                             |
| ------------------------- | --------------------------------------------------------------------- |
| LLM provider              | OpenAI                                                                |
| Primary model             | GPT-5 mini                                                            |
| Embedding model           | text-embedding-3-small                                                |
| Configuration             | Environment variables + application configuration                     |
| Deployment                | Docker containers                                                     |
| Backend                   | FastAPI                                                               |
| Database                  | PostgreSQL + pgvector                                                 |
| Product search            | Structured SQL + vector similarity                                    |
| Logging                   | Structured application logs                                           |
| Evaluation                | Curated shopping query/product evaluation dataset                     |
| Guardrails                | Prompt rules + tool permissions + confirmation                        |
| Latency monitoring        | Request, retrieval, tool, and LLM execution times                     |
| Token/cost tracking       | Store model usage/token metadata                                      |
| Error handling            | Timeouts, retries for transient LLM failures, safe fallback responses |
| Recommendation evaluation | Relevance, constraint satisfaction, groundedness                      |
| Observability             | Request ID + tool execution tracing                                   |

---

# 7. Safety and Shopping Guardrails

Because the assistant can potentially perform actions that affect the user's shopping account, v1 should include explicit boundaries.

### Low-risk actions

Can execute automatically:

* Search products
* Retrieve product information
* Compare products
* Retrieve reviews
* Search knowledge base
* View cart
* Add items to wishlist

### Medium-risk actions

May require confirmation:

* Add product to cart
* Remove product from cart
* Change quantity
* Apply coupon

### High-risk actions

Always require explicit confirmation:

* Checkout
* Place order
* Purchase product
* Use payment method
* Cancel an order
* Change important delivery information

The LLM must never independently authorize a financial transaction.

---

# 8. Example End-to-End Architecture

```text
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │   FastAPI API    │
                         └────────┬─────────┘
                                  ↓
                    ┌──────────────────────────┐
                    │   Shopping AI Agent      │
                    │      GPT-5 mini          │
                    └────────────┬─────────────┘
                                 ↓
              ┌──────────────────┼──────────────────┐
              ↓                  ↓                  ↓
       Product Search       RAG Search        Shopping Tools
              ↓                  ↓                  ↓
       Structured SQL       pgvector DB       Cart / Wishlist
              │                  │              Checkout
              └──────────┬───────┴──────────────┘
                         ↓
                 PostgreSQL Database
                         ↓
               Product + User + Cart
               + Embedding Data
```

---

# Final v1 Principle

Build a **single FastAPI-based AI Shopping Assistant** with:

* PostgreSQL + pgvector
* Product/catalog RAG
* Structured product filtering
* One controlled tool-using agent
* GPT-5 mini
* text-embedding-3-small
* Product comparison and recommendation
* Cart/wishlist tools
* Explicit confirmation for high-risk shopping actions
* Lightweight LLMOps
* Docker deployment
* Evaluation and observability

### Explicitly Out of Scope for v1

* Fine-tuning
* Distillation
* MCP
* Multi-agent orchestration
* Separate vector database
* Complex model routing
* Large LLMOps platforms
* Autonomous purchasing without confirmation

**Core architecture:**

> **RAG + Structured Product Search + Single Agent + Controlled Shopping Tools + Guardrails**

The important difference from your **AI Service Desk** project is that this system should use a **hybrid retrieval strategy**: vector search for semantic understanding and PostgreSQL filtering for exact shopping constraints such as **price, brand, category, rating, availability, and specifications**. This will make the project much stronger technically than a simple "LLM + product database" chatbot.
