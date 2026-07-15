import chromadb
from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("1. Sistem ve Modeller Yükleniyor... (Lütfen bekleyin)")
    config = Configuration(app_name="my-first-local-rag-app") 
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    # A) Vektör Modelini Yükle (Soruları sayılara çevirmek için)
    print(" - Qwen Vektör modeli yükleniyor...")
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.load()
    embed_client = embed_model.get_embedding_client()
    
    # B) Sohbet Modelini Yükle (Cevapları üretmek için)
    print(" - Phi-3.5 Sohbet modeli yükleniyor...")
    chat_model = manager.catalog.get_model("phi-3.5-mini")
    chat_model.load()
    chat_client = chat_model.get_chat_client()
    
    # C) Veritabanına Bağlan
    print("2. Vektör Veritabanına bağlanılıyor...")
    db_client = chromadb.PersistentClient(path="./vektor_db")
    collection = db_client.get_collection(name="pdf_notlarim")
    
    print("\n" + "="*60)
    print("🚀 RAG SİSTEMİ HAZIR! Vize notlarına soru sorabilirsin.")
    print("   (Çıkmak için 'q' veya 'çıkış' yazın)")
    print("="*60)
    
    while True:
        soru = input("\nSen: ")
        if soru.lower() in ['q', 'çıkış']:
            print("Sistem kapatılıyor. İyi çalışmalar!")
            break
            
        print("\nBelgeler aranıyor...")
        
        # 1. Kullanıcının sorusunu vektöre çevir
        soru_vektoru = embed_client.generate_embeddings(inputs=[soru]).data[0].embedding
        
        # 2. Veritabanında sorunun koordinatına en yakın 3 PDF sayfasını bul
        sonuclar = collection.query(
            query_embeddings=[soru_vektoru],
            n_results=3
        )
        
        # Bulunan sayfaları birleştir
        bulunan_metinler = sonuclar['documents'][0]
        baglam = "\n\n--- YENİ SAYFA ---\n\n".join(bulunan_metinler)
        
        # 3. Phi-3.5'e bağlamı ve soruyu ver (Prompt Engineering)
        sistem_mesaji = f"""Sen uzman bir asistan ve yazılım mühendisliği eğitmenisin. 
Sana verilen aşağıdaki PDF notlarına dayanarak kullanıcının sorusunu cevapla.
Eğer sorunun cevabı notlarda yoksa, uydurma ve "Bu bilgiyi notlarda bulamadım" de.

PDF NOTLARI (Bağlam):
{baglam}
"""
        
        messages = [
            {"role": "system", "content": sistem_mesaji},
            {"role": "user", "content": soru}
        ]
        
        print("Yapay Zeka Okuyor ve Düşünüyor...")
        # İşte bulduğumuz doğru komut!
        response = chat_client.complete_chat(messages=messages)
        
        # Sadece içeriği yazdır
        print("\n🤖 Yapay Zeka:")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    main()

