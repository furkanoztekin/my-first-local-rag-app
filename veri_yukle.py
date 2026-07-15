import chromadb
from pypdf import PdfReader
from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("1. Qwen Vektör Modeli VRAM'e yükleniyor...")
    config = Configuration(app_name="my-first-local-rag-app") 
    FoundryLocalManager.initialize(config)
    
    embed_model = FoundryLocalManager.instance.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    print("2. ChromaDB yerel veritabanı başlatılıyor...")
    # Proje klasöründe 'vektor_db' adında gizli bir klasör oluşturup verileri oraya yazar
    db_client = chromadb.PersistentClient(path="./vektor_db")
    
    # Veritabanında bir "Koleksiyon" (Tablo) oluşturuyoruz
    collection = db_client.get_or_create_collection(name="pdf_notlarim")

    print("3. PDF dosyası okunuyor ve sayfalara (chunk) ayrılıyor...")
    reader = PdfReader("test.pdf")
    
    sayfa_metinleri = []
    sayfa_idleri = []
    
    # Sayfaları tek tek okuyup listeye ekliyoruz
    for i, page in enumerate(reader.pages):
        metin = page.extract_text()
        if metin and len(metin.strip()) > 10: # Boş veya çok kısa sayfaları atla
            sayfa_metinleri.append(metin)
            sayfa_idleri.append(f"sayfa_{i+1}")
            
    print(f"Toplam {len(sayfa_metinleri)} dolu sayfa tespit edildi.")
    
    print("4. Sayfalar Qwen modeli ile 1024 boyutlu vektörlere dönüştürülüyor...")
    # 'inputs' parametresi ile listeyi topluca (batch) modele gönderiyoruz
    response = embed_client.generate_embeddings(inputs=sayfa_metinleri)
    
    # API'den dönen OpenAI formatındaki nesneden sadece sayıları çıkarıyoruz
    vektorler = [item.embedding for item in response.data]
    
    print("5. Vektörler ve orijinal metinler ChromaDB'ye kaydediliyor...")
    collection.add(
        ids=sayfa_idleri,
        embeddings=vektorler,
        documents=sayfa_metinleri
    )
    
    print("\n✅ İŞLEM TAMAM! Dosya başarıyla vektör veritabanına işlendi.")

if __name__ == "__main__":
    main()