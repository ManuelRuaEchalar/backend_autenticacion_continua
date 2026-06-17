import os
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.bucket_name = os.environ.get("AWS_S3_BUCKET_NAME")
        
        # Boto3 uses environment variables automatically (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
        try:
            self.s3_client = boto3.client('s3')
        except Exception as e:
            logger.error(f"Error inicializando cliente S3: {e}")
            self.s3_client = None

    def download_start_model(self, local_path: str, object_name: str = "startModel.keras") -> bool:
        """
        Descarga el modelo inicial desde S3 al path local especificado.
        """
        if not self.s3_client or not self.bucket_name:
            logger.warning("Cliente S3 no inicializado o bucket no configurado. Omitiendo descarga.")
            return False

        try:
            logger.info(f"Descargando {object_name} desde bucket {self.bucket_name} a {local_path}...")
            self.s3_client.download_file(self.bucket_name, object_name, local_path)
            logger.info("Descarga exitosa.")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.warning(f"El objeto {object_name} no existe en S3.")
            else:
                logger.error(f"Error descargando {object_name} de S3: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado descargando de S3: {e}")
            return False

    def upload_checkpoint(self, local_path: str, object_name: str) -> bool:
        """
        Sube el modelo guardado localmente a S3.
        """
        if not self.s3_client or not self.bucket_name:
            logger.warning("Cliente S3 no inicializado o bucket no configurado. Omitiendo subida.")
            return False

        try:
            logger.info(f"Subiendo {local_path} a S3 como {object_name}...")
            self.s3_client.upload_file(local_path, self.bucket_name, object_name)
            logger.info("Subida exitosa.")
            return True
        except Exception as e:
            logger.error(f"Error subiendo {local_path} a S3: {e}")
            return False
