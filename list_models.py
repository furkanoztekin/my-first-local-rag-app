from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("🚀 SDK Katalog Taraması Başlatılıyor...\n")
    
    # 1. SDK Başlatma
    config = Configuration(app_name="my-first-local-rag-app")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    # 2. Katalogdan tüm modelleri çek
    catalog = manager.catalog
    models = catalog.list_models()
    
    print(f"📦 Katalogda toplam {len(models)} varyant bulundu.\n")
    print("-" * 70)
    print(f"{'TÜR':<10} | {'ALİAS (Kısa Ad)':<25} | {'VARYANT ID (Gerçek İsim)'}")
    print("-" * 70)
    
    gpu_model_sayisi = 0
    cpu_model_sayisi = 0

    # 3. Modelleri listele ve GPU/CPU olarak sınıflandır
    for m in models:
        # Varyant ID'sinin içinde cuda veya gpu kelimesi geçiyor mu?
        if "cuda" in m.id.lower() or "gpu" in m.id.lower():
            tur = "🟢 GPU"
            gpu_model_sayisi += 1
        else:
            tur = "⚪ CPU"
            cpu_model_sayisi += 1
            
        print(f"{tur:<10} | {m.alias:<25} | {m.id}")

    print("-" * 70)
    print(f"📊 ÖZET: {gpu_model_sayisi} GPU modeli, {cpu_model_sayisi} CPU modeli listelendi.")

if __name__ == "__main__":
    main()