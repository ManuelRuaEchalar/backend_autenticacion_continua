import datetime
from typing import Dict, List, Any
# Se importa el MÓDULO, no `SessionLocal` por valor: init_db() anula el
# sessionmaker cuando la base de datos resulta inalcanzable, y una copia local
# se quedaría con el objeto vivo, pagando el timeout TCP en cada ronda.
from app import database
from app.database import TrainingRound
import logging

logger = logging.getLogger(__name__)

class FederationService:
    """
    Servicio encargado de mantener el historial y estado de las rondas
    de Aprendizaje Federado.
    
    Responsabilidad única: registrar y exponer métricas (progreso).
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._evaluations: List[Dict[str, Any]] = []

    def _save_to_db(self, db_record: TrainingRound):
        if database.SessionLocal:
            try:
                with database.SessionLocal() as db:
                    db.add(db_record)
                    db.commit()
            except Exception as e:
                logger.error(f"Error al guardar registro en la base de datos: {e}")

    def record_round(self, round_number: int, n_clients: int, total_samples: int, aggregated_metrics: Dict[str, Any]) -> None:
        """
        Registra los resultados agregados (loss, accuracy, etc.) después del entrenamiento (fit).
        """
        record = {
            "round": round_number,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_clients": n_clients,
            "total_samples": total_samples,
            "metrics": aggregated_metrics,
            "type": "fit"
        }
        self._history.append(record)
        
        # Guardar en base de datos
        db_record = TrainingRound(
            round_number=round_number,
            n_clients=n_clients,
            total_samples=total_samples,
            metrics=aggregated_metrics,
            record_type="fit"
        )
        self._save_to_db(db_record)

    def record_evaluation(self, round_number: int, weighted_eer: float, n_clients: int) -> None:
        """
        Registra los resultados de la evaluación agregada.
        """
        record = {
            "round": round_number,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "weighted_eer": weighted_eer,
            "n_clients": n_clients,
            "type": "evaluate"
        }
        self._evaluations.append(record)
        
        # Guardar en base de datos
        db_record = TrainingRound(
            round_number=round_number,
            n_clients=n_clients,
            record_type="evaluate",
            weighted_eer=weighted_eer
        )
        self._save_to_db(db_record)

    def get_status(self) -> Dict[str, Any]:
        """
        Devuelve un resumen del estado actual de la federación.
        """
        last_round = self._history[-1] if self._history else None
        last_eval = self._evaluations[-1] if self._evaluations else None
        
        return {
            "total_rounds_completed": len(self._history),
            "last_round": last_round,
            "last_evaluation": last_eval
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Devuelve el historial completo de rondas de entrenamiento.
        """
        return self._history
